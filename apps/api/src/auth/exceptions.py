"""Authentication exceptions."""

from fastapi import HTTPException, status


class AuthenticationError(HTTPException):
    """Base authentication error."""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class InvalidTokenError(AuthenticationError):
    """Invalid or expired token."""

    def __init__(self, detail: str = "Invalid or expired token"):
        super().__init__(detail=detail)


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password."""

    def __init__(self, detail: str = "Invalid username or password"):
        super().__init__(detail=detail)


class TokenExpiredError(AuthenticationError):
    """Token has expired."""

    def __init__(self, detail: str = "Token has expired"):
        super().__init__(detail=detail)
