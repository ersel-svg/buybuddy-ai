"""Authentication service using BuyBuddy API."""

import httpx
from typing import Optional
from pydantic import BaseModel

from config import settings
from auth.exceptions import InvalidCredentialsError, InvalidTokenError


class UserInfo(BaseModel):
    """User information from BuyBuddy API."""
    username: str
    token: str


class AuthService:
    """Authentication service using BuyBuddy Legacy API."""

    def __init__(self):
        self.base_url = settings.buybuddy_api_url
        # Cache validated tokens to avoid repeated API calls
        self._token_cache: dict[str, UserInfo] = {}

    async def login(self, username: str, password: str) -> UserInfo:
        """
        Login with BuyBuddy credentials and return token.

        Flow:
        1. POST /user/sign_in -> get passphrase
        2. POST /user/sign_in/token -> get token
        """
        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Sign in to get passphrase
            try:
                login_resp = await client.post(
                    f"{self.base_url}/user/sign_in",
                    json={
                        "user_name": username,
                        "password": password,
                    },
                    headers={"Content-Type": "application/json"},
                )

                # API returns 201 on successful login
                if login_resp.status_code not in (200, 201):
                    raise InvalidCredentialsError()

                passphrase = login_resp.json().get("passphrase")
                if not passphrase:
                    raise InvalidCredentialsError("Login failed - no passphrase returned")

            except httpx.RequestError:
                raise InvalidCredentialsError("Unable to connect to authentication server")

            # Step 2: Exchange passphrase for token
            try:
                token_resp = await client.post(
                    f"{self.base_url}/user/sign_in/token",
                    json={"passphrase": passphrase},
                    headers={"Content-Type": "application/json"},
                )

                # API may return 200 or 201
                if token_resp.status_code not in (200, 201):
                    raise InvalidCredentialsError("Failed to get token")

                token = token_resp.json().get("token")
                if not token:
                    raise InvalidCredentialsError("Login failed - no token returned")

            except httpx.RequestError:
                raise InvalidCredentialsError("Unable to connect to authentication server")

        # Cache the token
        user_info = UserInfo(username=username, token=token)
        self._token_cache[token] = user_info

        return user_info

    async def validate_token(self, token: str) -> UserInfo:
        """
        Validate a token by making a test request to BuyBuddy API.
        Returns user info if valid.
        """
        # Check cache first
        if token in self._token_cache:
            return self._token_cache[token]

        # Validate by making a request to BuyBuddy API
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get(
                    f"{self.base_url}/ai/product",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    params={"limit": 1},  # Minimal request
                )

                if resp.status_code == 401:
                    raise InvalidTokenError()

                if resp.status_code != 200:
                    raise InvalidTokenError(f"Token validation failed: {resp.status_code}")

            except httpx.RequestError:
                raise InvalidTokenError("Unable to validate token")

        # Token is valid - cache it with unknown username
        user_info = UserInfo(username="authenticated_user", token=token)
        self._token_cache[token] = user_info

        return user_info

    def invalidate_token(self, token: str) -> None:
        """Remove token from cache (logout)."""
        self._token_cache.pop(token, None)

    def clear_cache(self) -> None:
        """Clear all cached tokens."""
        self._token_cache.clear()


# Singleton instance
auth_service = AuthService()
