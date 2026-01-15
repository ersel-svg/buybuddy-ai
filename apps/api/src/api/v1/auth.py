"""Authentication API endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.service import auth_service, UserInfo
from auth.dependencies import get_current_user


router = APIRouter()


class LoginRequest(BaseModel):
    """Login request body."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response with token."""
    token: str
    username: str
    message: str = "Login successful"


class LogoutResponse(BaseModel):
    """Logout response."""
    message: str = "Logout successful"


class UserResponse(BaseModel):
    """Current user response."""
    username: str
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
    return UserResponse(username=user.username)
