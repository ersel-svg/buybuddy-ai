"""Product Matcher API router for matching uploaded product lists with system products."""

import io
import csv
import uuid
import logging
import httpx
from typing import Optional, List, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.supabase import supabase_service
from auth.dependencies import get_current_user
from fastapi import Depends
from config import settings

logger = logging.getLogger(__name__)

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
    all_rows: Optional[List[dict]] = None  # All rows (optional, for large files)


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


def parse_csv_file(content: bytes, filename: str, include_all_rows: bool = False) -> ParsedFileResponse:
    """
    Parse CSV file content.
    
    Args:
        content: File content as bytes
        filename: Original filename
        include_all_rows: If True, include all rows in response (for smaller files)
    
    Returns:
        ParsedFileResponse with columns, preview, and optionally all rows
    """
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

    # Normalize row values (preserve leading zeros)
    normalized_rows = []
    for row in rows:
        normalized_row = {}
        for key, value in row.items():
            if value is not None:
                # Preserve leading zeros by keeping as string
                normalized_row[key] = str(value).strip()
            else:
                normalized_row[key] = ""
        normalized_rows.append(normalized_row)

    columns = list(normalized_rows[0].keys()) if normalized_rows else []
    preview = normalized_rows[:5]

    return ParsedFileResponse(
        file_name=filename,
        columns=columns,
        total_rows=len(normalized_rows),
        preview=preview,
        all_rows=normalized_rows if include_all_rows and len(normalized_rows) <= 1000 else None
    )


def normalize_cell_value(cell_value: Any, preserve_leading_zeros: bool = False) -> str:
    """
    Normalize cell value to string, preserving important formatting.
    
    Args:
        cell_value: The cell value from Excel/CSV
        preserve_leading_zeros: If True, preserve leading zeros (important for barcodes)
    
    Returns:
        Normalized string value
    """
    if cell_value is None:
        return ""
    
    # For numeric values, preserve leading zeros if needed
    if isinstance(cell_value, (int, float)):
        # Check if it's a whole number that might have leading zeros
        if isinstance(cell_value, float) and cell_value.is_integer():
            # Convert to int first to remove .0
            int_value = int(cell_value)
            if preserve_leading_zeros:
                # Return as string to preserve any leading zeros that might be in original
                return str(int_value)
            return str(int_value)
        elif isinstance(cell_value, int):
            return str(cell_value)
        else:
            # Float with decimals
            return str(cell_value)
    
    # Already a string
    return str(cell_value).strip()


def parse_excel_file(content: bytes, filename: str, include_all_rows: bool = False) -> ParsedFileResponse:
    """
    Parse Excel file content.
    
    Args:
        content: File content as bytes
        filename: Original filename
        include_all_rows: If True, include all rows in response (for smaller files)
    
    Returns:
        ParsedFileResponse with columns, preview, and optionally all rows
    """
    try:
        import openpyxl
        from io import BytesIO

        workbook = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
        sheet = workbook.active

        rows_data = []
        headers = []

        for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
            if row_idx == 0:
                # Header row - clean column names
                headers = []
                for i, cell in enumerate(row):
                    if cell is not None:
                        header = str(cell).strip()
                        if header:
                            headers.append(header)
                        else:
                            headers.append(f"Column_{i}")
                    else:
                        headers.append(f"Column_{i}")
            else:
                # Data row
                row_dict = {}
                for col_idx, cell in enumerate(row):
                    if col_idx < len(headers):
                        # Normalize cell value (preserve leading zeros for potential barcodes)
                        # We'll preserve leading zeros by default - the matching logic will handle normalization
                        cell_str = normalize_cell_value(cell, preserve_leading_zeros=True)
                        row_dict[headers[col_idx]] = cell_str
                
                # Only add non-empty rows
                if any(v for v in row_dict.values() if v):
                    rows_data.append(row_dict)

        workbook.close()

        if not rows_data:
            raise HTTPException(status_code=400, detail="Excel file is empty or has no data rows")

        return ParsedFileResponse(
            file_name=filename,
            columns=headers,
            total_rows=len(rows_data),
            preview=rows_data[:5],
            all_rows=rows_data if include_all_rows and len(rows_data) <= 1000 else None
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl library not installed for Excel support")
    except Exception as e:
        logger.error(f"Excel parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")


async def send_bulk_scan_request_slack_notification(
    created_count: int,
    skipped_count: int,
    requester_name: str,
    requester_email: str,
    source_file: Optional[str] = None,
    sample_barcodes: Optional[List[str]] = None
) -> bool:
    """Send a single Slack notification for bulk scan request creation."""
    if not settings.slack_webhook_url:
        logger.warning("Slack webhook URL not configured, skipping notification")
        return False

    if created_count == 0:
        return False

    try:
        # Build sample barcodes text
        sample_text = ""
        if sample_barcodes and len(sample_barcodes) > 0:
            shown = sample_barcodes[:5]
            sample_text = ", ".join(f"`{b}`" for b in shown)
            if len(sample_barcodes) > 5:
                sample_text += f" ... and {len(sample_barcodes) - 5} more"

        # Build source file text
        source_text = f"from _{source_file}_" if source_file else ""

        message = {
            "username": "BuyBuddy AI",
            "icon_url": "https://qvyxpfcwfktxnaeavkxx.supabase.co/storage/v1/object/public/scan-request-images/branding/bb-logomark.png",
            "attachments": [
                {
                    "color": "#8b5cf6",  # Purple color for bulk
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*:package: Bulk Scan Requests Created*\n\n*{created_count}* new scan requests created {source_text}"
                            }
                        },
                        {
                            "type": "divider"
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Requested by:*\n{requester_name}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Email:*\n{requester_email}"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        # Add sample barcodes if available
        if sample_text:
            message["attachments"][0]["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Sample barcodes:*\n{sample_text}"
                }
            })

        # Add skipped info if any
        if skipped_count > 0:
            message["attachments"][0]["blocks"].append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f":warning: {skipped_count} duplicates skipped (already have pending requests)"
                    }
                ]
            })

        # Add timestamp
        message["attachments"][0]["blocks"].append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":clock1: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                }
            ]
        })

        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.slack_webhook_url,
                json=message,
                timeout=10.0
            )
            if response.status_code != 200:
                logger.error(f"Slack error response: {response.text}")
            return response.status_code == 200

    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}", exc_info=True)
        return False


def normalize_value_for_matching(value: Any, field_type: str) -> Optional[str]:
    """
    Normalize a value for matching, preserving important formatting.
    
    Args:
        value: The value to normalize
        field_type: Type of field ('barcode', 'sku', 'upc', 'ean', 'short_code' for numeric IDs,
                    or 'text' for text fields)
    
    Returns:
        Normalized string value, or None if empty/invalid
    """
    if value is None:
        return None
    
    # Convert to string
    str_value = str(value).strip()
    
    if not str_value:
        return None
    
    # For numeric identifier fields (barcode, sku, upc, ean, short_code)
    # Preserve leading zeros and exact format
    numeric_fields = {'barcode', 'sku', 'upc', 'ean', 'short_code'}
    
    if field_type in numeric_fields:
        # For numeric IDs, preserve the exact string representation
        # Don't convert to lower case to preserve case-sensitive formats
        # But normalize whitespace
        return str_value
    
    # For text fields, normalize to lowercase for case-insensitive matching
    return str_value.lower()


async def get_all_products_for_matching() -> dict:
    """
    Fetch all products and their identifiers for matching.
    Returns dict with lookup tables for fast matching.
    
    Uses lists for values to handle duplicates (multiple products with same value).
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

    # Build lookup tables for all fields - use lists to handle duplicates
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
                # Normalize based on field type
                normalized = normalize_value_for_matching(value, field)
                if normalized:
                    # Use lists to handle multiple products with same value
                    if normalized not in lookup[f'by_{field}']:
                        lookup[f'by_{field}'][normalized] = []
                    lookup[f'by_{field}'][normalized].append(product)

    # Index products by identifiers
    products_by_id = {p['id']: p for p in products}

    for identifier in identifiers:
        product = products_by_id.get(identifier['product_id'])
        if not product:
            continue

        id_type = identifier['identifier_type']
        id_value = identifier['identifier_value']
        
        # Normalize identifier value based on type
        normalized = normalize_value_for_matching(id_value, id_type)
        if not normalized:
            continue

        # Use lists to handle duplicates
        lookup_key = f'by_{id_type}'
        if normalized not in lookup[lookup_key]:
            lookup[lookup_key][normalized] = []
        lookup[lookup_key][normalized].append(product)

    return lookup


# ===========================================
# Endpoints
# ===========================================


@router.post("/upload", response_model=ParsedFileResponse)
async def upload_file(
    file: UploadFile = File(...),
    include_all_rows: bool = Query(False, description="Include all rows in response (for files <1000 rows)")
) -> ParsedFileResponse:
    """
    Upload and parse a CSV or Excel file.
    Returns columns and preview rows for mapping.
    
    Args:
        file: The file to upload
        include_all_rows: If True and file is small (<1000 rows), include all rows in response
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
    try:
        if filename_lower.endswith('.csv'):
            return parse_csv_file(content, file.filename, include_all_rows=include_all_rows)
        else:
            return parse_excel_file(content, file.filename, include_all_rows=include_all_rows)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


@router.post("/match", response_model=MatchResponse)
async def match_products(request: MatchRequest) -> MatchResponse:
    """
    Match uploaded rows against system products using the provided mapping.
    Uses priority-based matching: first match wins.

    For large datasets (1000+ rows), use /match/async endpoint instead.
    """
    if not request.rows:
        raise HTTPException(status_code=400, detail="No rows to match")

    if not request.mapping_config.match_rules:
        raise HTTPException(status_code=400, detail="No match rules provided")

    # Threshold for async processing
    ASYNC_MATCH_THRESHOLD = 1000

    # For large batches, recommend async
    if len(request.rows) >= ASYNC_MATCH_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Too many rows ({len(request.rows)}). Use /match/async endpoint for {ASYNC_MATCH_THRESHOLD}+ rows"
        )

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

            # Normalize value for lookup based on target field type
            normalized_value = normalize_value_for_matching(source_value, rule.target_field)
            if not normalized_value:
                continue

            # Get the appropriate lookup table
            lookup_key = field_to_lookup.get(rule.target_field)
            if not lookup_key:
                continue

            lookup_table = lookup.get(lookup_key, {})

            # Try to find match - handle both single product and list of products
            if normalized_value in lookup_table:
                products_list = lookup_table[normalized_value]
                
                # If it's a list (duplicates), take the first one
                # In future, we could add logic to choose the best match
                if isinstance(products_list, list) and len(products_list) > 0:
                    found_product = products_list[0]
                elif isinstance(products_list, dict):
                    # Legacy format (single product dict)
                    found_product = products_list
                else:
                    found_product = products_list
                
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


@router.post("/match/async")
async def match_products_async(request: MatchRequest):
    """
    Async version: Match products as a background job.

    Use this for large datasets (1000+ rows) to avoid memory issues and timeouts.
    Returns job_id for progress tracking.
    """
    if not request.rows:
        raise HTTPException(status_code=400, detail="No rows to match")

    if not request.mapping_config.match_rules:
        raise HTTPException(status_code=400, detail="No match rules provided")

    # Convert match rules to serializable format
    match_rules = [
        {
            "source_column": r.source_column,
            "target_field": r.target_field,
            "priority": r.priority
        }
        for r in request.mapping_config.match_rules
    ]

    # Create job record
    job = await supabase_service.create_job({
        "type": "local_bulk_product_matcher",
        "config": {
            "rows": request.rows,
            "match_rules": match_rules,
        }
    })

    return {
        "job_id": job["id"],
        "status": "pending",
        "message": f"Matching job queued for {len(request.rows)} rows",
    }


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

        # Send single Slack notification for bulk creation
        sample_barcodes = [r["barcode"] for r in records_to_insert[:10]]
        await send_bulk_scan_request_slack_notification(
            created_count=created_count,
            skipped_count=len(skipped_barcodes),
            requester_name=request.requester_name,
            requester_email=request.requester_email,
            source_file=request.source_file,
            sample_barcodes=sample_barcodes
        )

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
