"""Product Matcher API router for matching uploaded product lists with system products."""

import io
import csv
import uuid
from typing import Optional, List, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.supabase import supabase_service
from auth.dependencies import get_current_user
from fastapi import Depends

router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================


class ParsedFileResponse(BaseModel):
    """Response from file upload/parse."""
    file_name: str
    columns: List[str]
    total_rows: int
    preview: List[dict]  # First 5 rows


class MatchRule(BaseModel):
    """A single match rule mapping source column to system field."""
    source_column: str
    target_field: str  # barcode, sku, upc, ean, short_code, product_name, brand_name
    priority: int


class MappingConfig(BaseModel):
    """Configuration for field mapping."""
    match_rules: List[MatchRule]


class MatchRequest(BaseModel):
    """Request to perform matching."""
    rows: List[dict]  # All parsed rows from file
    mapping_config: MappingConfig


class MatchedProduct(BaseModel):
    """A matched product from the system."""
    id: str
    barcode: str
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    category: Optional[str] = None
    status: str


class MatchedItem(BaseModel):
    """A successfully matched item."""
    source_row: dict
    product: MatchedProduct
    matched_by: str  # Which field matched


class UnmatchedItem(BaseModel):
    """An unmatched item."""
    source_row: dict


class MatchSummary(BaseModel):
    """Summary of matching results."""
    total: int
    matched_count: int
    unmatched_count: int
    match_rate: float


class MatchResponse(BaseModel):
    """Response from matching operation."""
    matched: List[MatchedItem]
    unmatched: List[UnmatchedItem]
    summary: MatchSummary


class BulkScanRequestItem(BaseModel):
    """A single item for bulk scan request creation."""
    barcode: str
    product_name: Optional[str] = None
    brand_name: Optional[str] = None


class BulkScanRequestCreate(BaseModel):
    """Request to create bulk scan requests."""
    items: List[BulkScanRequestItem]
    requester_name: str
    requester_email: str
    source_file: Optional[str] = None
    notes: Optional[str] = None


class BulkScanRequestResponse(BaseModel):
    """Response from bulk scan request creation."""
    created_count: int
    skipped_count: int
    skipped_barcodes: List[str]


class ExportRequest(BaseModel):
    """Request for CSV export."""
    items: List[dict]
    columns: List[str]


# ===========================================
# Helper Functions
# ===========================================


def parse_csv_file(content: bytes, filename: str) -> ParsedFileResponse:
    """Parse CSV file content."""
    try:
        # Try to detect encoding
        text = content.decode('utf-8-sig')  # Handle BOM
    except UnicodeDecodeError:
        try:
            text = content.decode('latin-1')
        except UnicodeDecodeError:
            text = content.decode('utf-8', errors='replace')

    # Detect delimiter
    sample = text[:2000]
    delimiter = ','
    if sample.count(';') > sample.count(','):
        delimiter = ';'
    elif sample.count('\t') > sample.count(','):
        delimiter = '\t'

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no data rows")

    columns = list(rows[0].keys()) if rows else []
    preview = rows[:5]

    return ParsedFileResponse(
        file_name=filename,
        columns=columns,
        total_rows=len(rows),
        preview=preview
    )


def parse_excel_file(content: bytes, filename: str) -> ParsedFileResponse:
    """Parse Excel file content."""
    try:
        import openpyxl
        from io import BytesIO

        workbook = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
        sheet = workbook.active

        rows_data = []
        headers = []

        for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
            if row_idx == 0:
                # Header row
                headers = [str(cell) if cell is not None else f"Column_{i}" for i, cell in enumerate(row)]
            else:
                # Data row
                row_dict = {}
                for col_idx, cell in enumerate(row):
                    if col_idx < len(headers):
                        row_dict[headers[col_idx]] = str(cell) if cell is not None else ""
                if any(row_dict.values()):  # Skip empty rows
                    rows_data.append(row_dict)

        workbook.close()

        if not rows_data:
            raise HTTPException(status_code=400, detail="Excel file is empty or has no data rows")

        return ParsedFileResponse(
            file_name=filename,
            columns=headers,
            total_rows=len(rows_data),
            preview=rows_data[:5]
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl library not installed for Excel support")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")


async def get_all_products_for_matching() -> dict:
    """
    Fetch all products and their identifiers for matching.
    Returns dict with lookup tables for fast matching.
    """
    # Get all products with all matchable fields
    products_result = supabase_service.client.table("products").select(
        "id, barcode, product_name, brand_name, sub_brand, category, "
        "variant_flavor, container_type, net_quantity, manufacturer_country, "
        "marketing_description, status"
    ).execute()

    products = products_result.data or []

    # Get all product identifiers
    identifiers_result = supabase_service.client.table("product_identifiers").select(
        "product_id, identifier_type, identifier_value"
    ).execute()

    identifiers = identifiers_result.data or []

    # All product fields that can be matched
    product_fields = [
        'barcode', 'product_name', 'brand_name', 'sub_brand', 'category',
        'variant_flavor', 'container_type', 'net_quantity', 'manufacturer_country',
        'marketing_description'
    ]

    # Build lookup tables for all fields
    lookup = {f'by_{field}': {} for field in product_fields}
    # Add identifier lookups
    lookup.update({
        'by_sku': {},
        'by_upc': {},
        'by_ean': {},
        'by_short_code': {},
    })

    # Index products by all fields
    for product in products:
        for field in product_fields:
            value = product.get(field)
            if value and isinstance(value, str) and value.strip():
                lookup[f'by_{field}'][value.strip().lower()] = product

    # Index products by identifiers
    products_by_id = {p['id']: p for p in products}

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

    return lookup


# ===========================================
# Endpoints
# ===========================================


@router.post("/upload", response_model=ParsedFileResponse)
async def upload_file(file: UploadFile = File(...)) -> ParsedFileResponse:
    """
    Upload and parse a CSV or Excel file.
    Returns columns and preview rows for mapping.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file type
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith('.csv') or filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported formats: .csv, .xlsx, .xls"
        )

    # Read file content
    content = await file.read()

    # Validate file size (max 10MB)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")

    # Parse based on file type
    if filename_lower.endswith('.csv'):
        return parse_csv_file(content, file.filename)
    else:
        return parse_excel_file(content, file.filename)


@router.post("/match", response_model=MatchResponse)
async def match_products(request: MatchRequest) -> MatchResponse:
    """
    Match uploaded rows against system products using the provided mapping.
    Uses priority-based matching: first match wins.
    """
    if not request.rows:
        raise HTTPException(status_code=400, detail="No rows to match")

    if not request.mapping_config.match_rules:
        raise HTTPException(status_code=400, detail="No match rules provided")

    # Sort rules by priority
    sorted_rules = sorted(request.mapping_config.match_rules, key=lambda r: r.priority)

    # Get all products for matching (single query)
    lookup = await get_all_products_for_matching()

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

    matched: List[MatchedItem] = []
    unmatched: List[UnmatchedItem] = []

    for row in request.rows:
        found_product = None
        matched_by = None

        # Try each rule in priority order
        for rule in sorted_rules:
            source_value = row.get(rule.source_column)
            if not source_value:
                continue

            # Normalize value for lookup
            normalized_value = str(source_value).strip().lower()
            if not normalized_value:
                continue

            # Get the appropriate lookup table
            lookup_key = field_to_lookup.get(rule.target_field)
            if not lookup_key:
                continue

            lookup_table = lookup.get(lookup_key, {})

            # Try to find match
            if normalized_value in lookup_table:
                found_product = lookup_table[normalized_value]
                matched_by = rule.target_field
                break

        if found_product:
            matched.append(MatchedItem(
                source_row=row,
                product=MatchedProduct(
                    id=found_product['id'],
                    barcode=found_product.get('barcode', ''),
                    product_name=found_product.get('product_name'),
                    brand_name=found_product.get('brand_name'),
                    category=found_product.get('category'),
                    status=found_product.get('status', 'pending')
                ),
                matched_by=matched_by
            ))
        else:
            unmatched.append(UnmatchedItem(source_row=row))

    total = len(request.rows)
    matched_count = len(matched)
    unmatched_count = len(unmatched)
    match_rate = (matched_count / total * 100) if total > 0 else 0

    return MatchResponse(
        matched=matched,
        unmatched=unmatched,
        summary=MatchSummary(
            total=total,
            matched_count=matched_count,
            unmatched_count=unmatched_count,
            match_rate=round(match_rate, 1)
        )
    )


@router.post("/create-scan-requests", response_model=BulkScanRequestResponse)
async def create_bulk_scan_requests(request: BulkScanRequestCreate) -> BulkScanRequestResponse:
    """
    Create scan requests for multiple unmatched products.
    Skips duplicates (existing pending/in_progress requests with same barcode).
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="No items provided")

    # Get all existing pending/in_progress scan requests
    existing_result = supabase_service.client.table("scan_requests").select(
        "barcode"
    ).in_("status", ["pending", "in_progress"]).execute()

    existing_barcodes = {r['barcode'].strip().lower() for r in (existing_result.data or []) if r.get('barcode')}

    created_count = 0
    skipped_barcodes = []

    # Prepare bulk insert
    records_to_insert = []

    for item in request.items:
        barcode = item.barcode.strip()
        if not barcode:
            continue

        # Check for duplicate
        if barcode.lower() in existing_barcodes:
            skipped_barcodes.append(barcode)
            continue

        # Build notes
        notes = request.notes or ""
        if request.source_file:
            notes = f"Bulk import from {request.source_file}. {notes}".strip()

        records_to_insert.append({
            "barcode": barcode,
            "product_name": item.product_name,
            "brand_name": item.brand_name,
            "notes": notes,
            "requester_name": request.requester_name,
            "requester_email": request.requester_email,
            "reference_images": [],
            "status": "pending"
        })

        # Add to existing set to avoid duplicates within same batch
        existing_barcodes.add(barcode.lower())

    # Bulk insert
    if records_to_insert:
        result = supabase_service.client.table("scan_requests").insert(records_to_insert).execute()
        created_count = len(result.data) if result.data else 0

    return BulkScanRequestResponse(
        created_count=created_count,
        skipped_count=len(skipped_barcodes),
        skipped_barcodes=skipped_barcodes
    )


@router.post("/export/matched")
async def export_matched_csv(request: ExportRequest) -> StreamingResponse:
    """Export matched items as CSV."""
    if not request.items:
        raise HTTPException(status_code=400, detail="No items to export")

    # Create CSV in memory
    output = io.StringIO()

    # Determine columns: source columns + matched product info
    source_columns = request.columns
    product_columns = ['matched_product_id', 'matched_barcode', 'matched_product_name', 'matched_brand', 'matched_by']
    all_columns = source_columns + product_columns

    writer = csv.DictWriter(output, fieldnames=all_columns)
    writer.writeheader()

    for item in request.items:
        row = {}
        # Source row data
        source_row = item.get('source_row', {})
        for col in source_columns:
            row[col] = source_row.get(col, '')

        # Matched product data
        product = item.get('product', {})
        row['matched_product_id'] = product.get('id', '')
        row['matched_barcode'] = product.get('barcode', '')
        row['matched_product_name'] = product.get('product_name', '')
        row['matched_brand'] = product.get('brand_name', '')
        row['matched_by'] = item.get('matched_by', '')

        writer.writerow(row)

    # Prepare response
    output.seek(0)
    content = output.getvalue()

    # Add BOM for Excel compatibility
    content_bytes = '\ufeff' + content

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"matched_products_{len(request.items)}_{timestamp}.csv"

    return StreamingResponse(
        iter([content_bytes]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.post("/export/unmatched")
async def export_unmatched_csv(request: ExportRequest) -> StreamingResponse:
    """Export unmatched items as CSV."""
    if not request.items:
        raise HTTPException(status_code=400, detail="No items to export")

    # Create CSV in memory
    output = io.StringIO()

    # Use provided columns
    writer = csv.DictWriter(output, fieldnames=request.columns)
    writer.writeheader()

    for item in request.items:
        source_row = item.get('source_row', item)
        row = {col: source_row.get(col, '') for col in request.columns}
        writer.writerow(row)

    # Prepare response
    output.seek(0)
    content = output.getvalue()

    # Add BOM for Excel compatibility
    content_bytes = '\ufeff' + content

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"unmatched_products_{len(request.items)}_{timestamp}.csv"

    return StreamingResponse(
        iter([content_bytes]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# All system fields available for matching
SYSTEM_FIELDS = [
    # Primary identifiers
    {"value": "barcode", "label": "Barcode", "description": "Product barcode"},
    {"value": "sku", "label": "SKU", "description": "Stock Keeping Unit"},
    {"value": "upc", "label": "UPC", "description": "Universal Product Code"},
    {"value": "ean", "label": "EAN", "description": "European Article Number"},
    {"value": "short_code", "label": "Short Code", "description": "Short product code"},
    # Product info fields
    {"value": "product_name", "label": "Product Name", "description": "Full product name"},
    {"value": "brand_name", "label": "Brand Name", "description": "Product brand"},
    {"value": "sub_brand", "label": "Sub Brand", "description": "Sub-brand or product line"},
    {"value": "category", "label": "Category", "description": "Product category"},
    {"value": "variant_flavor", "label": "Variant/Flavor", "description": "Product variant or flavor"},
    {"value": "container_type", "label": "Container Type", "description": "Package type (bottle, can, box, etc.)"},
    {"value": "net_quantity", "label": "Net Quantity", "description": "Product size/weight"},
    {"value": "manufacturer_country", "label": "Country", "description": "Manufacturing country"},
    {"value": "marketing_description", "label": "Description", "description": "Product marketing description"},
]


@router.get("/system-fields")
async def get_system_fields() -> List[dict]:
    """Get list of available system fields for matching."""
    return SYSTEM_FIELDS
