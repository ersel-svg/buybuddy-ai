"""FastAPI dependencies for authentication."""

from typing import Optional
from fastapi import Depends, Header

from auth.service import auth_service, UserInfo
from auth.exceptions import AuthenticationError, InvalidTokenError


async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> UserInfo:
    """
    Dependency that validates the Authorization header and returns user info.

    Usage:
        @router.get("/protected")
        async def protected_route(user: UserInfo = Depends(get_current_user)):
            return {"username": user.username}
    """
    if not authorization:
        raise AuthenticationError("Missing Authorization header")

    # Extract token from "Bearer <token>" format
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError("Invalid Authorization header format. Use: Bearer <token>")

    token = parts[1]

    if not token:
        raise AuthenticationError("Empty token")

    # Validate token with BuyBuddy API
    try:
        user_info = await auth_service.validate_token(token)
        return user_info
    except InvalidTokenError:
        raise
    except Exception as e:
        raise AuthenticationError(f"Token validation failed: {str(e)}")


async def get_optional_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[UserInfo]:
    """
    Dependency that optionally validates the Authorization header.
    Returns None if no valid token is provided (instead of raising an error).

    Usage:
        @router.get("/public-or-private")
        async def mixed_route(user: Optional[UserInfo] = Depends(get_optional_user)):
            if user:
                return {"authenticated": True, "username": user.username}
            return {"authenticated": False}
    """
    if not authorization:
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1]
    if not token:
        return None

    try:
        return await auth_service.validate_token(token)
    except Exception:
        return None
