"""
Handler for bulk updating products.

This handler processes product updates in batches with progress tracking,
handling both product fields and identifier fields.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


# Fields that can be updated in the products table
PRODUCT_FIELDS = {
    "product_name",
    "brand_name",
    "sub_brand",
    "category",
    "variant_flavor",
    "container_type",
    "net_quantity",
    "manufacturer_country",
    "marketing_description",
    "pack_configuration",
    "nutrition_facts",
    "claims",
    "visibility_score",
}

# Fields that are stored in product_identifiers table
IDENTIFIER_FIELDS = {"sku", "upc", "ean", "short_code"}


@job_registry.register
class BulkUpdateProductsHandler(BaseJobHandler):
    """
    Handler for bulk updating products.

    Config:
        updates: list[dict] - Each dict has {product_id, fields}
        mode: str - "strict" or "lenient" (default: "lenient")

    Result:
        updated: int - Number of products updated
        failed: int - Number of products that failed
        total: int - Total products processed
        errors: list[dict] - Failed updates with product_id and error
    """

    job_type = "local_bulk_update_products"
    BATCH_SIZE = 50

    def validate_config(self, config: dict) -> str | None:
        if not config.get("updates"):
            return "updates is required"
        if not isinstance(config["updates"], list):
            return "updates must be a list"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        updates = config["updates"]
        mode = config.get("mode", "lenient")
        total = len(updates)

        if total == 0:
            return {
                "updated": 0,
                "failed": 0,
                "total": 0,
                "errors": [],
                "message": "No products to update",
            }

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=total,
        ))

        # Process updates
        updated = 0
        failed = 0
        errors = []

        for batch_num, batch in enumerate(chunks(updates, self.BATCH_SIZE)):
            for update in batch:
                product_id = update.get("product_id")
                fields = update.get("fields", {})

                if not product_id:
                    failed += 1
                    errors.append({"product_id": None, "error": "Missing product_id"})
                    continue

                try:
                    # Separate product fields and identifier fields
                    product_fields = {
                        k: v for k, v in fields.items()
                        if k in PRODUCT_FIELDS and v is not None
                    }
                    identifier_fields = {
                        k: v for k, v in fields.items()
                        if k in IDENTIFIER_FIELDS and v is not None
                    }

                    # Update product fields
                    if product_fields:
                        await supabase_service.update_product_fields(
                            product_id,
                            product_fields
                        )

                    # Update identifier fields
                    if identifier_fields:
                        await supabase_service.upsert_product_identifiers(
                            product_id,
                            identifier_fields
                        )

                    updated += 1

                except Exception as e:
                    failed += 1
                    errors.append({
                        "product_id": product_id,
                        "error": str(e),
                    })

                    if mode == "strict":
                        raise ValueError(f"Strict mode: Update failed for {product_id}: {e}")

            # Update progress
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = calculate_progress(processed, total) * 0.95

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Updating products... ({processed}/{total})",
                processed=processed,
                total=total,
            ))

        return {
            "updated": updated,
            "failed": failed,
            "total": total,
            "errors": errors[:20] if errors else [],  # Limit errors to 20
            "message": f"Updated {updated} products" + (f", {failed} failed" if failed else ""),
        }
