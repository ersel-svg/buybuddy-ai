"""
Handler for bulk product matching operations.

This handler processes product matching in batches with progress tracking,
using pagination to avoid loading all products into memory.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import calculate_progress


@job_registry.register
class BulkProductMatcherHandler(BaseJobHandler):
    """
    Handler for bulk product matching with pagination.

    Config:
        rows: list[dict] - Rows to match
        match_rules: list[dict] - Match rules with priority
            Each rule: {source_column, target_field, priority}

    Result:
        matched: int - Number of rows matched
        unmatched: int - Number of rows unmatched
        total: int - Total rows processed
        match_rate: float - Match percentage
        matched_items: list[dict] - Matched items (limited to 1000)
        unmatched_items: list[dict] - Unmatched items (limited to 1000)
    """

    job_type = "local_bulk_product_matcher"
    PAGE_SIZE = 1000  # Pagination size for product loading

    def validate_config(self, config: dict) -> str | None:
        if not config.get("rows"):
            return "rows is required"
        if not config.get("match_rules"):
            return "match_rules is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        rows = config["rows"]
        match_rules = sorted(config["match_rules"], key=lambda r: r["priority"])
        total = len(rows)

        if total == 0:
            return {
                "matched": 0,
                "unmatched": 0,
                "total": 0,
                "match_rate": 0.0,
                "matched_items": [],
                "unmatched_items": [],
                "message": "No rows to match",
            }

        update_progress(JobProgress(
            progress=5,
            current_step="Loading products for matching...",
            processed=0,
            total=total,
        ))

        # Build lookup tables with pagination (avoid loading all products)
        lookup = await self._build_lookup_tables_paginated()

        update_progress(JobProgress(
            progress=20,
            current_step="Starting matching...",
            processed=0,
            total=total,
        ))

        # Map target fields to lookup tables
        field_to_lookup = {
            # Product fields
            'barcode': 'by_barcode',
            'product_name': 'by_product_name',
            'brand_name': 'by_brand_name',
            'sub_brand': 'by_sub_brand',
            'category': 'by_category',
            'variant_flavor': 'by_variant_flavor',
            'container_type': 'by_container_type',
            'net_quantity': 'by_net_quantity',
            'manufacturer_country': 'by_manufacturer_country',
            'marketing_description': 'by_marketing_description',
            # Identifier fields
            'sku': 'by_sku',
            'upc': 'by_upc',
            'ean': 'by_ean',
            'short_code': 'by_short_code',
        }

        matched_items = []
        unmatched_items = []
        matched_count = 0
        unmatched_count = 0

        # Process rows
        for idx, row in enumerate(rows):
            found_product = None
            matched_by = None

            # Try each rule in priority order
            for rule in match_rules:
                source_value = row.get(rule["source_column"])
                if not source_value:
                    continue

                # Normalize value for lookup
                normalized_value = str(source_value).strip().lower()
                if not normalized_value:
                    continue

                # Get the appropriate lookup table
                lookup_key = field_to_lookup.get(rule["target_field"])
                if not lookup_key:
                    continue

                lookup_table = lookup.get(lookup_key, {})

                # Try to find match
                if normalized_value in lookup_table:
                    found_product = lookup_table[normalized_value]
                    matched_by = rule["target_field"]
                    break

            # Store results (limit to 1000 items to avoid memory issues)
            if found_product:
                matched_count += 1
                if len(matched_items) < 1000:
                    matched_items.append({
                        "source_row": row,
                        "product": {
                            "id": found_product['id'],
                            "barcode": found_product.get('barcode', ''),
                            "product_name": found_product.get('product_name'),
                            "brand_name": found_product.get('brand_name'),
                            "category": found_product.get('category'),
                            "status": found_product.get('status', 'pending')
                        },
                        "matched_by": matched_by
                    })
            else:
                unmatched_count += 1
                if len(unmatched_items) < 1000:
                    unmatched_items.append({
                        "source_row": row
                    })

            # Update progress (20-95% range)
            if (idx + 1) % 100 == 0 or idx == total - 1:
                progress = 20 + calculate_progress(idx + 1, total) * 0.75
                update_progress(JobProgress(
                    progress=int(progress),
                    current_step=f"Matching... ({idx + 1}/{total})",
                    processed=idx + 1,
                    total=total,
                ))

        match_rate = (matched_count / total * 100) if total > 0 else 0

        return {
            "matched": matched_count,
            "unmatched": unmatched_count,
            "total": total,
            "match_rate": round(match_rate, 1),
            "matched_items": matched_items,
            "unmatched_items": unmatched_items,
            "message": f"Matched {matched_count}/{total} rows ({match_rate:.1f}%)",
        }

    async def _build_lookup_tables_paginated(self) -> dict:
        """Build lookup tables using pagination to avoid memory issues."""
        product_fields = [
            'barcode', 'product_name', 'brand_name', 'sub_brand', 'category',
            'variant_flavor', 'container_type', 'net_quantity', 'manufacturer_country',
            'marketing_description'
        ]

        # Initialize lookup tables
        lookup = {f'by_{field}': {} for field in product_fields}
        lookup.update({
            'by_sku': {},
            'by_upc': {},
            'by_ean': {},
            'by_short_code': {},
        })

        # Paginate through products
        offset = 0
        while True:
            result = supabase_service.client.table("products").select(
                "id, barcode, product_name, brand_name, sub_brand, category, "
                "variant_flavor, container_type, net_quantity, manufacturer_country, "
                "marketing_description, status"
            ).range(offset, offset + self.PAGE_SIZE - 1).execute()

            products = result.data or []
            if not products:
                break

            # Index products by all fields
            for product in products:
                for field in product_fields:
                    value = product.get(field)
                    if value and isinstance(value, str) and value.strip():
                        lookup[f'by_{field}'][value.strip().lower()] = product

            offset += self.PAGE_SIZE
            if len(products) < self.PAGE_SIZE:
                break

        # Load product identifiers (also paginated)
        offset = 0
        products_by_id = {}  # Store products we've seen

        while True:
            identifiers_result = supabase_service.client.table("product_identifiers").select(
                "product_id, identifier_type, identifier_value"
            ).range(offset, offset + self.PAGE_SIZE - 1).execute()

            identifiers = identifiers_result.data or []
            if not identifiers:
                break

            # Get unique product IDs from this batch
            product_ids = list(set(i['product_id'] for i in identifiers))

            # Fetch products for these identifiers if not already loaded
            new_product_ids = [pid for pid in product_ids if pid not in products_by_id]
            if new_product_ids:
                products_result = supabase_service.client.table("products").select(
                    "id, barcode, product_name, brand_name, category, status"
                ).in_("id", new_product_ids).execute()

                for product in (products_result.data or []):
                    products_by_id[product['id']] = product

            # Index identifiers
            for identifier in identifiers:
                product = products_by_id.get(identifier['product_id'])
                if not product:
                    continue

                id_type = identifier['identifier_type']
                id_value = identifier['identifier_value'].strip().lower()

                if id_type == 'sku':
                    lookup['by_sku'][id_value] = product
                elif id_type == 'upc':
                    lookup['by_upc'][id_value] = product
                elif id_type == 'ean':
                    lookup['by_ean'][id_value] = product
                elif id_type == 'short_code':
                    lookup['by_short_code'][id_value] = product

            offset += self.PAGE_SIZE
            if len(identifiers) < self.PAGE_SIZE:
                break

        return lookup
