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

    async def _ensure_token(self) -> str:
        """Ensure we have a valid token."""
        if not self._token:
            self._token = await self._login()
        return self._token

    async def get_products(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Fetch products with video URLs from BuyBuddy API.

        Returns list of products with:
        - barcode
        - video_url
        - video_id
        - name (if available)
        """
        token = await self._ensure_token()

        async with httpx.AsyncClient(timeout=60) as client:
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
        for product in products_list[:limit]:
            video = product.get("video", {})
            video_product = video.get("product", {})

            barcode = video_product.get("upc")
            video_url = video.get("media_url")
            video_id = video.get("id")
            processed = product.get("processed", False)

            if barcode and video_url:
                result.append({
                    "barcode": barcode,
                    "video_url": video_url,
                    "video_id": video_id,
                    "name": video_product.get("name"),
                    "processed": processed,
                })

        return result

    async def get_unprocessed_products(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get only unprocessed products."""
        all_products = await self.get_products(limit=limit * 2)
        return [p for p in all_products if not p.get("processed")][:limit]


# Singleton instance
buybuddy_service = BuybuddyService()
