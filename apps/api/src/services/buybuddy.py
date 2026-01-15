"""BuyBuddy Legacy API client for fetching products with video URLs."""

import httpx
from typing import Any, Optional
from config import settings


class BuybuddyService:
    """Client for BuyBuddy Legacy API."""

    def __init__(self):
        self.base_url = settings.buybuddy_api_url
        self.username = settings.buybuddy_username
        self.password = settings.buybuddy_password
        self._token: Optional[str] = None

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

        Returns:
            Dict with:
            - items: List of cutout images [{id, cutout_image_url, upc}]
            - page: Current page
            - page_size: Page size
            - has_more: Whether there are more pages
        """
        token = await self._ensure_token()

        # Build query params
        params = {
            "page": page,
            "page_size": page_size,
        }
        if sort_field:
            params["field"] = sort_field
            params["sort"] = sort_order

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                f"{self.base_url}/ai/basket_image/cutout",
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
                    f"{self.base_url}/ai/basket_image/cutout",
                    params=params,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )

            resp.raise_for_status()
            data = resp.json()

        # API returns array directly
        items = data if isinstance(data, list) else data.get("data", [])

        # Transform to our format
        result_items = []
        for item in items:
            result_items.append({
                "external_id": item.get("id"),
                "image_url": item.get("cutout_image_url"),
                "predicted_upc": item.get("upc"),
            })

        return {
            "items": result_items,
            "page": page,
            "page_size": page_size,
            "has_more": len(items) == page_size,  # If full page, likely more
        }

    async def get_all_cutout_images(
        self,
        max_pages: Optional[int] = None,
        page_size: int = 100,
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """
        Fetch all cutout images (paginated internally).

        Args:
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


# Singleton instance
buybuddy_service = BuybuddyService()
