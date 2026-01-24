"""Product Bulk Update API router for bulk updating product information from Excel/CSV."""

import logging
from typing import Optional, List, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from services.supabase import supabase_service
from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Constants
# ===========================================

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

# All updatable fields
ALL_UPDATABLE_FIELDS = PRODUCT_FIELDS | IDENTIFIER_FIELDS

# Field validation rules
FIELD_VALIDATIONS = {
    "product_name": {"max_length": 500},
    "brand_name": {"max_length": 200},
    "sub_brand": {"max_length": 200},
    "category": {"max_length": 100},
    "variant_flavor": {"max_length": 200},
    "container_type": {"max_length": 100},
    "net_quantity": {"max_length": 50},
    "manufacturer_country": {"max_length": 100},
    "marketing_description": {"max_length": 2000},
    "visibility_score": {"min": 0, "max": 100},
    "sku": {"max_length": 100},
    "upc": {"max_length": 50},
    "ean": {"max_length": 50},
    "short_code": {"max_length": 50},
}


# ===========================================
# Schemas
# ===========================================


class FieldMapping(BaseModel):
    """Mapping from source column to target field."""
    source_column: str
    target_field: str  # One of ALL_UPDATABLE_FIELDS


class PreviewRequest(BaseModel):
    """Request for bulk update preview."""
    rows: List[dict]
    identifier_column: str  # Column used to match products (usually barcode)
    field_mappings: List[FieldMapping]


class ProductChange(BaseModel):
    """Details of changes for a single product."""
    row_index: int
    product_id: str
    barcode: str
    current_values: dict
    new_values: dict
    product_field_changes: List[str]
    identifier_field_changes: List[str]


class NotFoundItem(BaseModel):
    """An item that couldn't be matched to any product."""
    row_index: int
    identifier_value: str
    source_row: dict


class ValidationError(BaseModel):
    """A validation error for a field."""
    row_index: int
    field: str
    value: Any
    error: str


class PreviewSummary(BaseModel):
    """Summary of preview results."""
    total_rows: int
    matched: int
    not_found: int
    validation_errors: int
    will_update: int


class PreviewResponse(BaseModel):
    """Response from preview operation."""
    matches: List[ProductChange]
    not_found: List[NotFoundItem]
    validation_errors: List[ValidationError]
    summary: PreviewSummary


class UpdateItem(BaseModel):
    """A single product update."""
    product_id: str
    fields: dict


class ExecuteRequest(BaseModel):
    """Request to execute bulk update."""
    updates: List[UpdateItem]
    mode: str = Field(default="lenient", pattern="^(strict|lenient)$")


class FailedUpdate(BaseModel):
    """A failed update."""
    product_id: str
    error: str


class ExecuteResponse(BaseModel):
    """Response from execute operation."""
    success: bool
    updated_count: int
    failed: List[FailedUpdate]
    execution_time_ms: int


class SystemField(BaseModel):
    """A system field that can be mapped."""
    id: str
    label: str
    group: str
    editable: bool


# ===========================================
# Helper Functions
# ===========================================


def validate_field_value(field: str, value: Any) -> Optional[str]:
    """Validate a field value. Returns error message or None if valid."""
    if value is None or value == "":
        return None  # Empty values are allowed (they won't update the field)

    rules = FIELD_VALIDATIONS.get(field, {})

    # String length validation
    if "max_length" in rules and isinstance(value, str):
        if len(value) > rules["max_length"]:
            return f"Value exceeds maximum length of {rules['max_length']}"

    # Numeric range validation
    if "min" in rules or "max" in rules:
        try:
            num_value = int(value) if isinstance(value, str) else value
            if "min" in rules and num_value < rules["min"]:
                return f"Value must be at least {rules['min']}"
            if "max" in rules and num_value > rules["max"]:
                return f"Value must be at most {rules['max']}"
        except (ValueError, TypeError):
            return "Value must be a number"

    # JSONB validation for structured fields
    if field in ("pack_configuration", "nutrition_facts"):
        if isinstance(value, str):
            try:
                import json
                json.loads(value)
            except json.JSONDecodeError:
                return "Value must be valid JSON"

    # Array validation for claims
    if field == "claims":
        if isinstance(value, str):
            # Accept comma-separated values
            pass
        elif not isinstance(value, list):
            return "Value must be a list or comma-separated string"

    return None


def normalize_value(field: str, value: Any) -> Any:
    """Normalize a value for the given field."""
    if value is None or value == "":
        return None

    # String fields - trim whitespace
    if field in PRODUCT_FIELDS - {"pack_configuration", "nutrition_facts", "claims", "visibility_score"}:
        return str(value).strip() if value else None

    # Integer fields
    if field == "visibility_score":
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    # JSONB fields
    if field in ("pack_configuration", "nutrition_facts"):
        if isinstance(value, str):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value

    # Array fields
    if field == "claims":
        if isinstance(value, str):
            return [c.strip() for c in value.split(",") if c.strip()]
        return value

    # Identifier fields
    if field in IDENTIFIER_FIELDS:
        return str(value).strip() if value else None

    return value


# ===========================================
# Endpoints
# ===========================================


@router.get("/system-fields", response_model=List[SystemField])
async def get_system_fields():
    """Get list of system fields available for mapping."""
    fields = [
        # Identifiers (barcode is not editable as it's the primary key)
        SystemField(id="barcode", label="Barcode", group="identifiers", editable=False),
        SystemField(id="sku", label="SKU", group="identifiers", editable=True),
        SystemField(id="upc", label="UPC", group="identifiers", editable=True),
        SystemField(id="ean", label="EAN", group="identifiers", editable=True),
        SystemField(id="short_code", label="Short Code", group="identifiers", editable=True),

        # Product Info
        SystemField(id="product_name", label="Product Name", group="product_info", editable=True),
        SystemField(id="brand_name", label="Brand Name", group="product_info", editable=True),
        SystemField(id="sub_brand", label="Sub Brand", group="product_info", editable=True),
        SystemField(id="category", label="Category", group="product_info", editable=True),
        SystemField(id="variant_flavor", label="Variant/Flavor", group="product_info", editable=True),
        SystemField(id="container_type", label="Container Type", group="product_info", editable=True),
        SystemField(id="net_quantity", label="Net Quantity", group="product_info", editable=True),
        SystemField(id="manufacturer_country", label="Country", group="product_info", editable=True),
        SystemField(id="marketing_description", label="Description", group="product_info", editable=True),

        # Structured
        SystemField(id="pack_configuration", label="Pack Config", group="structured", editable=True),
        SystemField(id="nutrition_facts", label="Nutrition Facts", group="structured", editable=True),
        SystemField(id="claims", label="Claims", group="structured", editable=True),

        # Quality
        SystemField(id="visibility_score", label="Visibility Score", group="quality", editable=True),
    ]
    return fields


@router.post("/preview", response_model=PreviewResponse)
async def preview_bulk_update(request: PreviewRequest):
    """
    Preview bulk update operation.

    Matches rows to products and shows what changes will be made.
    """
    logger.info(f"Preview bulk update: {len(request.rows)} rows, {len(request.field_mappings)} mappings")

    # Validate field mappings
    for mapping in request.field_mappings:
        if mapping.target_field not in ALL_UPDATABLE_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target field: {mapping.target_field}"
            )

    # Extract all identifier values from rows (exact match, preserve case)
    identifier_values = []
    for row in request.rows:
        value = row.get(request.identifier_column)
        if value:
            identifier_values.append(str(value).strip())

    # Fetch all products matching these identifiers
    products = await supabase_service.get_products_by_barcodes(identifier_values)

    # Build lookup table for fast matching (exact barcode match)
    product_lookup: dict[str, dict] = {}
    for product in products:
        barcode = product.get("barcode", "").strip()
        if barcode:
            product_lookup[barcode] = product

    # Fetch existing identifiers for matched products
    product_ids = [p["id"] for p in products]
    existing_identifiers = await supabase_service.get_identifiers_for_products(product_ids)

    # Build identifier lookup: product_id -> {sku: value, upc: value, ...}
    identifier_lookup: dict[str, dict[str, str]] = {}
    for identifier in existing_identifiers:
        pid = identifier["product_id"]
        if pid not in identifier_lookup:
            identifier_lookup[pid] = {}
        identifier_lookup[pid][identifier["identifier_type"]] = identifier["identifier_value"]

    # Process each row
    matches: List[ProductChange] = []
    not_found: List[NotFoundItem] = []
    validation_errors: List[ValidationError] = []

    for row_index, row in enumerate(request.rows):
        identifier_value = row.get(request.identifier_column)
        if not identifier_value:
            not_found.append(NotFoundItem(
                row_index=row_index,
                identifier_value="",
                source_row=row
            ))
            continue

        normalized_identifier = str(identifier_value).strip()
        product = product_lookup.get(normalized_identifier)

        if not product:
            not_found.append(NotFoundItem(
                row_index=row_index,
                identifier_value=str(identifier_value),
                source_row=row
            ))
            continue

        # Product found - calculate changes
        product_id = product["id"]
        current_identifiers = identifier_lookup.get(product_id, {})

        current_values: dict = {}
        new_values: dict = {}
        product_field_changes: List[str] = []
        identifier_field_changes: List[str] = []
        row_has_errors = False

        for mapping in request.field_mappings:
            source_value = row.get(mapping.source_column)
            target_field = mapping.target_field

            # Skip empty values
            if source_value is None or source_value == "":
                continue

            # Validate
            error = validate_field_value(target_field, source_value)
            if error:
                validation_errors.append(ValidationError(
                    row_index=row_index,
                    field=target_field,
                    value=source_value,
                    error=error
                ))
                row_has_errors = True
                continue

            # Normalize value
            normalized_value = normalize_value(target_field, source_value)
            if normalized_value is None:
                continue

            # Get current value
            if target_field in IDENTIFIER_FIELDS:
                current_value = current_identifiers.get(target_field)
            else:
                current_value = product.get(target_field)

            # Check if value actually changed
            if str(current_value or "") != str(normalized_value):
                current_values[target_field] = current_value
                new_values[target_field] = normalized_value

                if target_field in IDENTIFIER_FIELDS:
                    identifier_field_changes.append(target_field)
                else:
                    product_field_changes.append(target_field)

        # Add to matches if there are actual changes and no errors
        if (product_field_changes or identifier_field_changes) and not row_has_errors:
            matches.append(ProductChange(
                row_index=row_index,
                product_id=product_id,
                barcode=product.get("barcode", ""),
                current_values=current_values,
                new_values=new_values,
                product_field_changes=product_field_changes,
                identifier_field_changes=identifier_field_changes
            ))

    # Calculate matched = total rows - not_found
    # This represents all products that were found in the database
    # (regardless of whether they have changes or not)
    matched_count = len(request.rows) - len(not_found)

    summary = PreviewSummary(
        total_rows=len(request.rows),
        matched=matched_count,
        not_found=len(not_found),
        validation_errors=len(validation_errors),
        will_update=len(matches)
    )

    return PreviewResponse(
        matches=matches,
        not_found=not_found,
        validation_errors=validation_errors,
        summary=summary
    )


@router.post("/execute", response_model=ExecuteResponse)
async def execute_bulk_update(request: ExecuteRequest):
    """
    Execute bulk update operation.

    Updates products with the provided changes.
    """
    import time
    start_time = time.time()

    logger.info(f"Execute bulk update: {len(request.updates)} updates, mode={request.mode}")

    if not request.updates:
        return ExecuteResponse(
            success=True,
            updated_count=0,
            failed=[],
            execution_time_ms=0
        )

    updated_count = 0
    failed: List[FailedUpdate] = []

    # Separate product fields and identifier fields
    for update in request.updates:
        try:
            product_fields = {
                k: v for k, v in update.fields.items()
                if k in PRODUCT_FIELDS and v is not None
            }
            identifier_fields = {
                k: v for k, v in update.fields.items()
                if k in IDENTIFIER_FIELDS and v is not None
            }

            # Update product fields
            if product_fields:
                await supabase_service.update_product_fields(
                    update.product_id,
                    product_fields
                )

            # Update identifier fields
            if identifier_fields:
                await supabase_service.upsert_product_identifiers(
                    update.product_id,
                    identifier_fields
                )

            updated_count += 1

        except Exception as e:
            logger.error(f"Failed to update product {update.product_id}: {e}")
            failed.append(FailedUpdate(
                product_id=update.product_id,
                error=str(e)
            ))
            if request.mode == "strict":
                # In strict mode, fail the entire operation
                raise HTTPException(
                    status_code=500,
                    detail=f"Update failed for product {update.product_id}: {e}"
                )

    execution_time_ms = int((time.time() - start_time) * 1000)

    return ExecuteResponse(
        success=len(failed) == 0,
        updated_count=updated_count,
        failed=failed,
        execution_time_ms=execution_time_ms
    )


ASYNC_UPDATE_THRESHOLD = 50


@router.post("/execute/async")
async def execute_bulk_update_async(request: ExecuteRequest):
    """
    Async version: Execute bulk update as a background job.

    Use this for large updates (>50 products).
    """
    from uuid import uuid4
    from datetime import datetime, timezone

    if len(request.updates) < ASYNC_UPDATE_THRESHOLD:
        # Use sync version for small batches
        return await execute_bulk_update(request)

    # Convert updates to serializable format
    updates_data = [
        {"product_id": u.product_id, "fields": u.fields}
        for u in request.updates
    ]

    # Create job record
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "type": "local_bulk_update_products",
        "status": "pending",
        "progress": 0,
        "config": {
            "updates": updates_data,
            "mode": request.mode,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("jobs").insert(job_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Bulk update job queued for {len(request.updates)} products",
    }
