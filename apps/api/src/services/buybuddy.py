"""BuyBuddy Legacy API client for fetching products with video URLs.

SOTA Features:
- Exponential backoff retry with jitter
- Connection pooling for better performance
- Automatic token refresh on 401
- Rate limiting awareness
"""

import asyncio
import random
import httpx
from typing import Any, Optional
from config import settings


class BuybuddyService:
    """Client for BuyBuddy Legacy API with SOTA retry and pooling."""

    # Retry configuration
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    MAX_DELAY = 30.0  # seconds

    def __init__(self):
        self.base_url = settings.buybuddy_api_url
        self.username = settings.buybuddy_username
        self.password = settings.buybuddy_password
        self._token: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """Make HTTP request with exponential backoff retry."""
        client = await self._get_client()
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = await client.request(method, url, **kwargs)

                # Handle 401 - refresh token and retry
                if resp.status_code == 401 and attempt < self.MAX_RETRIES - 1:
                    self._token = await self._login()
                    if "headers" in kwargs:
                        kwargs["headers"]["Authorization"] = f"Bearer {self._token}"
                    continue

                # Handle rate limiting (429)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    continue

                resp.raise_for_status()
                return resp

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    # Exponential backoff with jitter
                    delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
                    jitter = random.uniform(0, delay * 0.1)
                    await asyncio.sleep(delay + jitter)
                    continue
                raise

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx) except 401, 429
                if 400 <= e.response.status_code < 500:
                    raise
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
                    await asyncio.sleep(delay)
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Request failed after retries")

    def is_configured(self) -> bool:
        """Check if credentials are configured."""
        return bool(self.username and self.password)

    async def _login(self) -> str:
        """Login and get auth token."""
        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Sign in
            login_resp = await client.post(
                f"{self.base_url}/user/sign_in",
                json={
                    "user_name": self.username,
                    "password": self.password,
                },
                headers={"Content-Type": "application/json"},
            )
            login_resp.raise_for_status()
            passphrase = login_resp.json().get("passphrase")

            # Step 2: Get token
            token_resp = await client.post(
                f"{self.base_url}/user/sign_in/token",
                json={"passphrase": passphrase},
                headers={"Content-Type": "application/json"},
            )
            token_resp.raise_for_status()
            return token_resp.json().get("token")

    async def _ensure_token(self, force_refresh: bool = False) -> str:
        """Ensure we have a valid token."""
        if not self._token or force_refresh:
            self._token = await self._login()
        return self._token

    def _clear_token(self):
        """Clear cached token (e.g., on 401 error)."""
        self._token = None

    async def get_products(self, limit: Optional[int] = None, only_unprocessed: bool = False) -> list[dict[str, Any]]:
        """
        Fetch products with video URLs from BuyBuddy API.

        Args:
            limit: Max products to return (None = all)
            only_unprocessed: If True, filter to only unprocessed products

        Returns list of products with:
        - barcode
        - video_url
        - video_id
        - name (if available)
        - processed (bool)
        """
        token = await self._ensure_token()

        # Longer timeout for large datasets
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.get(
                f"{self.base_url}/ai/product",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            raw_data = resp.json()

        products_list = raw_data.get("data", [])

        # Transform to our format
        result = []
        for product in products_list:
            video = product.get("video", {})
            video_product = video.get("product", {})

            barcode = video_product.get("upc")
            video_url = video.get("media_url")
            video_id = video.get("id")
            processed = product.get("processed", False)

            # Skip processed if only_unprocessed is set
            if only_unprocessed and processed:
                continue

            if barcode and video_url:
                result.append({
                    "barcode": barcode,
                    "video_url": video_url,
                    "video_id": video_id,
                    "name": video_product.get("name"),
                    "processed": processed,
                })

            # Stop if we've reached the limit
            if limit and len(result) >= limit:
                break

        return result

    async def get_unprocessed_products(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Get only unprocessed products."""
        return await self.get_products(limit=limit, only_unprocessed=True)

    async def get_cutout_images(
        self,
        page: int = 1,
        page_size: int = 100,
        sort_field: Optional[str] = None,
        sort_order: str = "asc",
        merchant_ids: Optional[list[int]] = None,
        inserted_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fetch cutout images from BuyBuddy API.

        These are AI-detected product cutout images from basket/shelf photos
        with predicted UPC codes.

        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            sort_field: Field to sort by (id, cutout_image_url, upc)
            sort_order: Sort direction (asc, desc)
            merchant_ids: List of merchant IDs to filter by (required for performance)
            inserted_at: Filter by insertion date (YYYY-MM-DD format)
            updated_at: Filter by update date (YYYY-MM-DD format)

        Returns:
            Dict with:
            - items: List of cutout images with merchant info and annotated UPC
            - page: Current page
            - page_size: Page size
            - page_count: Total number of pages
            - has_more: Whether there are more pages
        """
        if not merchant_ids or len(merchant_ids) == 0:
            raise ValueError("merchant_ids is required and must contain at least one ID")

        token = await self._ensure_token()

        # Build query params - merchant_id[] needs special handling
        params: list[tuple[str, Any]] = [
            ("page", page),
            ("page_size", page_size),
        ]
        if sort_field:
            params.append(("field", sort_field))
            params.append(("sort", sort_order))

        # Add merchant_id[] params (multiple values with same key)
        for mid in merchant_ids:
            params.append(("merchant_id[]", mid))

        # Add date filters
        if inserted_at:
            params.append(("inserted_at", inserted_at))
        if updated_at:
            params.append(("updated_at", updated_at))

        # Use retry-enabled request
        resp = await self._request_with_retry(
            "GET",
            f"{self.base_url}/ai/basket_image/cutout",
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        data = resp.json()

        # New API returns {page_count, data: [...]}
        page_count = data.get("page_count", 0) if isinstance(data, dict) else 0
        items = data.get("data", []) if isinstance(data, dict) else data

        # Transform to our format with new fields
        result_items = []
        for item in items:
            result_items.append({
                "external_id": item.get("id"),
                "image_url": item.get("cutout_image_url"),
                "predicted_upc": item.get("upc"),
                "merchant": item.get("merchant"),
                "row_index": item.get("row_index"),
                "column_index": item.get("column_index"),
                "annotated_upc": item.get("annotated_upc"),
            })

        return {
            "items": result_items,
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "has_more": page < page_count,
        }

    async def get_all_cutout_images(
        self,
        merchant_ids: list[int],
        max_pages: Optional[int] = None,
        page_size: int = 100,
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """
        Fetch all cutout images (paginated internally).

        Args:
            merchant_ids: List of merchant IDs to filter by (required)
            max_pages: Maximum pages to fetch (None for all)
            page_size: Items per page
            sort_order: Sort direction - "desc" for newest first, "asc" for oldest first

        Returns:
            List of all cutout images
        """
        all_items = []
        page = 1

        while True:
            result = await self.get_cutout_images(
                page=page,
                page_size=page_size,
                sort_field="id",
                sort_order=sort_order,
                merchant_ids=merchant_ids,
            )

            all_items.extend(result["items"])

            if not result["has_more"]:
                break

            if max_pages and page >= max_pages:
                break

            page += 1

            # Small delay to avoid rate limits
            import asyncio
            await asyncio.sleep(0.1)

        return all_items


    # ===========================================
    # Merchants
    # ===========================================

    async def get_merchants(
        self,
        all_merchant: bool = True,
        merchant_ids: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch merchants from BuyBuddy API.

        Args:
            all_merchant: If True, returns all merchants. If False, returns only user's merchant.
            merchant_ids: Filter by specific merchant IDs (optional).

        Returns:
            List of merchants with id, name, created_at, updated_at.
        """
        token = await self._ensure_token()

        # Build query params
        params: list[tuple[str, Any]] = []
        if all_merchant:
            params.append(("all_merchant", "true"))
        if merchant_ids:
            for mid in merchant_ids:
                params.append(("merchant_id[]", mid))

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.base_url}/merchant",
                params=params if params else None,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )

            # Retry with fresh token on 401
            if resp.status_code == 401:
                token = await self._ensure_token(force_refresh=True)
                resp = await client.get(
                    f"{self.base_url}/merchant",
                    params=params if params else None,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )

            resp.raise_for_status()
            data = resp.json()

        # Response is a list of merchants
        merchants = data if isinstance(data, list) else []

        return [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "created_at": m.get("created_at"),
                "updated_at": m.get("updated_at"),
            }
            for m in merchants
        ]

    # ===========================================
    # Object Detection - Evaluation Images
    # ===========================================

    async def get_evaluation_images(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        store_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
        is_annotated: Optional[bool] = None,
        is_approved: Optional[bool] = None,
    ) -> dict[str, Any]:
        """
        Fetch evaluation/basket images from BuyBuddy API for OD training.

        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            store_id: Filter by store ID
            limit: Max items to return (default 50)
            offset: Pagination offset (default 0)
            is_annotated: Filter by annotation status (optional)
            is_approved: Filter by approval status (optional)

        Returns:
            Dict with:
            - items: List of evaluation images
            - total_count: Total count
            - limit: Limit used
            - offset: Offset used
            - has_more: Whether there are more pages
        """
        token = await self._ensure_token()

        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if store_id is not None:
            params["store_id"] = store_id
        if is_annotated is not None:
            params["is_annotated"] = is_annotated
        if is_approved is not None:
            params["is_approved"] = is_approved

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                f"{self.base_url}/basket/evaluation/images",
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )

            # Retry with fresh token on 401
            if resp.status_code == 401:
                token = await self._ensure_token(force_refresh=True)
                resp = await client.get(
                    f"{self.base_url}/basket/evaluation/images",
                    params=params,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )

            resp.raise_for_status()
            data = resp.json()

        # Parse response
        images = data.get("images", [])
        total_count = data.get("total_count", len(images))

        # Transform to our format
        result_items = []
        for item in images:
            result_items.append({
                "buybuddy_image_id": str(item.get("image_id")),
                "image_url": item.get("image_url"),
                "image_type": item.get("image_type"),
                "inserted_at": item.get("inserted_at"),
                "basket_id": item.get("basket_id"),
                "basket_identifier": item.get("basket_identifier"),
                "merchant_id": item.get("merchant_id"),
                "merchant_name": item.get("merchant_name"),
                "store_id": item.get("store_id"),
                "store_name": item.get("store_name"),
                "store_code": item.get("store_code"),
                "is_annotated": item.get("is_annotated", False),
                "is_approved": item.get("is_approved", False),
                "annotation_id": item.get("annotation_id"),
                "in_datasets": item.get("in_datasets", []),
            })

        return {
            "items": result_items,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(images)) < total_count,
        }

    async def preview_od_sync(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        store_id: Optional[int] = None,
        is_annotated: Optional[bool] = None,
        is_approved: Optional[bool] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Preview what images would be synced from BuyBuddy.

        Returns count and sample of images that would be imported.
        """
        result = await self.get_evaluation_images(
            start_date=start_date,
            end_date=end_date,
            store_id=store_id,
            is_annotated=is_annotated,
            is_approved=is_approved,
            limit=limit,
            offset=0,
        )

        return {
            "total_available": result.get("total_count", len(result["items"])),
            "sample_count": len(result["items"]),
            "sample_images": result["items"],
            "filters_applied": {
                "start_date": start_date,
                "end_date": end_date,
                "store_id": store_id,
                "is_annotated": is_annotated,
                "is_approved": is_approved,
            }
        }

    async def sync_od_images(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        store_id: Optional[int] = None,
        is_annotated: Optional[bool] = None,
        is_approved: Optional[bool] = None,
        max_images: Optional[int] = None,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fetch all evaluation images for OD sync.

        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            store_id: Filter by store
            is_annotated: Filter by annotation status
            is_approved: Filter by approval status
            max_images: Maximum images to fetch
            batch_size: Items per batch

        Returns:
            List of all images matching criteria
        """
        all_items = []
        offset = 0

        while True:
            result = await self.get_evaluation_images(
                start_date=start_date,
                end_date=end_date,
                store_id=store_id,
                is_annotated=is_annotated,
                is_approved=is_approved,
                limit=batch_size,
                offset=offset,
            )

            all_items.extend(result["items"])

            # Check limits
            if max_images and len(all_items) >= max_images:
                all_items = all_items[:max_images]
                break

            if not result["has_more"]:
                break

            offset += batch_size

            # Rate limiting
            import asyncio
            await asyncio.sleep(0.1)

        return all_items


# Singleton instance
buybuddy_service = BuybuddyService()
