"""Authentication API endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.service import auth_service, UserInfo
from auth.dependencies import get_current_user
from services.buybuddy import buybuddy_service


router = APIRouter()


class LoginRequest(BaseModel):
    """Login request body."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response with token."""
    token: str
    username: str
    user_id: Optional[int] = None
    message: str = "Login successful"


class LogoutResponse(BaseModel):
    """Logout response."""
    message: str = "Logout successful"


class UserResponse(BaseModel):
    """Current user response."""
    username: str
    user_id: Optional[int] = None
    authenticated: bool = True


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """
    Login with BuyBuddy credentials.

    Returns a token that should be included in the Authorization header
    for all subsequent requests as: `Authorization: Bearer <token>`
    """
    user_info = await auth_service.login(
        username=request.username,
        password=request.password,
    )

    return LoginResponse(
        token=user_info.token,
        username=user_info.username,
        user_id=user_info.user_id,
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(user: UserInfo = Depends(get_current_user)) -> LogoutResponse:
    """
    Logout and invalidate the current token.

    Removes the token from the server-side cache.
    """
    auth_service.invalidate_token(user.token)
    return LogoutResponse()


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: UserInfo = Depends(get_current_user)) -> UserResponse:
    """
    Get current authenticated user information.

    Use this endpoint to verify that a token is valid.
    """
    return UserResponse(username=user.username, user_id=user.user_id)


@router.get("/stores")
async def get_user_stores(user: UserInfo = Depends(get_current_user)):
    """
    Get stores assigned to the current authenticated user.

    Returns a list of stores the user has access to.
    """
    if not user.user_id:
        raise HTTPException(
            status_code=400,
            detail="User ID not available. Please log in again.",
        )

    try:
        stores = await buybuddy_service.get_user_stores(user.user_id)
        return stores
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user stores: {str(e)}",
        )
