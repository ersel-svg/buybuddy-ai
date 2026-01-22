"""
Simple in-memory cache with TTL for filter options and other frequently accessed data.

This provides a lightweight caching layer without requiring Redis.
For production at scale, consider migrating to Redis or another distributed cache.
"""

import hashlib
import json
import logging
from typing import Any, Optional

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ===========================================
# Cache Configuration
# ===========================================

# Filter options cache: 30 second TTL, max 100 entries
# This caches the filter options response which is expensive to compute
_filter_cache: TTLCache = TTLCache(maxsize=100, ttl=30)

# Product count cache: 60 second TTL, max 50 entries
# Caches total product counts for pagination
_count_cache: TTLCache = TTLCache(maxsize=50, ttl=60)


# ===========================================
# Cache Key Generation
# ===========================================


def _generate_cache_key(prefix: str, params: dict) -> str:
    """
    Generate a deterministic cache key from parameters.

    Uses MD5 hash of sorted JSON to ensure consistent keys
    regardless of parameter order.
    """
    # Remove None values for consistent keys
    clean_params = {k: v for k, v in params.items() if v is not None}

    # Sort and serialize
    sorted_json = json.dumps(clean_params, sort_keys=True, default=str)
    param_hash = hashlib.md5(sorted_json.encode()).hexdigest()[:16]

    return f"{prefix}:{param_hash}"


# ===========================================
# Filter Options Cache
# ===========================================


def get_filter_options_cached(
    status: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    sub_brand: Optional[str] = None,
    exclude_dataset_id: Optional[str] = None,
    **kwargs,
) -> Optional[dict]:
    """
    Get cached filter options if available.

    Returns None if cache miss, otherwise returns the cached result.
    """
    cache_key = _generate_cache_key("filter_options", {
        "status": status,
        "category": category,
        "brand": brand,
        "sub_brand": sub_brand,
        "exclude_dataset_id": exclude_dataset_id,
        **kwargs,
    })

    result = _filter_cache.get(cache_key)
    if result is not None:
        logger.debug(f"Cache hit for filter_options: {cache_key}")
    return result


def set_filter_options_cached(
    result: dict,
    status: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    sub_brand: Optional[str] = None,
    exclude_dataset_id: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Cache filter options result.
    """
    cache_key = _generate_cache_key("filter_options", {
        "status": status,
        "category": category,
        "brand": brand,
        "sub_brand": sub_brand,
        "exclude_dataset_id": exclude_dataset_id,
        **kwargs,
    })

    _filter_cache[cache_key] = result
    logger.debug(f"Cached filter_options: {cache_key}")


def invalidate_filter_options_cache() -> int:
    """
    Invalidate all filter options cache entries.

    Call this when products are created, updated, or deleted.
    Returns the number of entries cleared.
    """
    count = len(_filter_cache)
    _filter_cache.clear()
    logger.info(f"Invalidated {count} filter options cache entries")
    return count


# ===========================================
# Product Count Cache
# ===========================================


def get_product_count_cached(filters_hash: str) -> Optional[int]:
    """Get cached product count for given filter hash."""
    return _count_cache.get(f"count:{filters_hash}")


def set_product_count_cached(filters_hash: str, count: int) -> None:
    """Cache product count for given filter hash."""
    _count_cache[f"count:{filters_hash}"] = count


def invalidate_product_count_cache() -> int:
    """Invalidate all product count cache entries."""
    count = len(_count_cache)
    _count_cache.clear()
    logger.info(f"Invalidated {count} product count cache entries")
    return count


# ===========================================
# Combined Invalidation
# ===========================================


def invalidate_all_product_caches() -> dict:
    """
    Invalidate all product-related caches.

    Call this when products are created, updated, or deleted.
    """
    filter_count = invalidate_filter_options_cache()
    count_count = invalidate_product_count_cache()

    return {
        "filter_options_cleared": filter_count,
        "counts_cleared": count_count,
    }


# ===========================================
# Cache Statistics
# ===========================================


def get_cache_stats() -> dict:
    """Get current cache statistics."""
    return {
        "filter_options": {
            "size": len(_filter_cache),
            "maxsize": _filter_cache.maxsize,
            "ttl": _filter_cache.ttl,
        },
        "product_counts": {
            "size": len(_count_cache),
            "maxsize": _count_cache.maxsize,
            "ttl": _count_cache.ttl,
        },
    }
