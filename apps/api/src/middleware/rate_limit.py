"""
Rate Limiting Middleware for BuyBuddy AI API.

Uses slowapi for request rate limiting with Redis backend (falls back to in-memory).

Configuration:
- General API: 120 requests/minute
- Workflow execution: 30 requests/minute (GPU-intensive)
- Heavy endpoints: 20 requests/minute

Usage in routes:
    from middleware.rate_limit import limiter

    @router.post("/run")
    @limiter.limit("30/minute")
    async def run_workflow(request: Request, ...):
        ...
"""

import logging
from typing import Optional, Callable

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from config import settings

logger = logging.getLogger(__name__)


def get_client_identifier(request: Request) -> str:
    """
    Get client identifier for rate limiting.

    Priority:
    1. X-API-Key header (for authenticated clients)
    2. X-Forwarded-For header (for proxied requests)
    3. Client IP address
    """
    # Check for API key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key[:16]}"  # Use first 16 chars for privacy

    # Check for forwarded IP (behind proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain (original client)
        return forwarded.split(",")[0].strip()

    # Fall back to direct client IP
    return get_remote_address(request)


def _get_storage_uri() -> str:
    """
    Get storage URI for rate limiter.

    Uses Redis if configured, otherwise falls back to in-memory storage.
    In-memory storage is suitable for single-instance deployments.
    """
    if settings.redis_url:
        logger.info(f"Rate limiter using Redis: {settings.redis_url[:30]}...")
        return settings.redis_url
    else:
        logger.info("Rate limiter using in-memory storage (no Redis configured)")
        return "memory://"


# Create limiter instance
limiter = Limiter(
    key_func=get_client_identifier,
    default_limits=[f"{settings.rate_limit_requests_per_minute}/minute"],
    storage_uri=_get_storage_uri(),
    strategy="fixed-window",  # or "moving-window" for smoother limiting
    headers_enabled=True,  # Add X-RateLimit-* headers to responses
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.

    Returns a JSON response with retry information.
    """
    from fastapi.responses import JSONResponse

    # Extract retry-after from exception if available
    retry_after = getattr(exc, "retry_after", 60)

    logger.warning(
        f"Rate limit exceeded for {get_client_identifier(request)} "
        f"on {request.method} {request.url.path}"
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Too many requests. Please retry after {retry_after} seconds.",
            "retry_after_seconds": retry_after,
            "limit": str(exc.detail) if hasattr(exc, "detail") else "unknown",
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(getattr(exc, "limit", "unknown")),
        },
    )


# Pre-defined rate limit decorators for common use cases
def workflow_rate_limit(func: Callable) -> Callable:
    """Rate limit decorator for workflow execution endpoints (GPU-intensive)."""
    return limiter.limit(f"{settings.rate_limit_workflow_runs_per_minute}/minute")(func)


def heavy_endpoint_limit(func: Callable) -> Callable:
    """Rate limit decorator for heavy computation endpoints."""
    return limiter.limit("20/minute")(func)


def light_endpoint_limit(func: Callable) -> Callable:
    """Rate limit decorator for lightweight read endpoints."""
    return limiter.limit("200/minute")(func)


# Export for use in main.py
__all__ = [
    "limiter",
    "RateLimitExceeded",
    "rate_limit_exceeded_handler",
    "SlowAPIMiddleware",
    "workflow_rate_limit",
    "heavy_endpoint_limit",
    "light_endpoint_limit",
]
