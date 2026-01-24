"""
Utility functions for local job handlers.
"""

from typing import Iterator, TypeVar

T = TypeVar("T")


def chunks(items: list[T], size: int) -> Iterator[list[T]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to split
        size: Maximum size of each chunk

    Yields:
        Lists of up to `size` items
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


def calculate_progress(processed: int, total: int) -> int:
    """
    Calculate progress percentage safely.

    Args:
        processed: Number of items processed
        total: Total number of items

    Returns:
        Progress percentage (0-100)
    """
    if total <= 0:
        return 0
    return min(int(processed / total * 100), 100)
