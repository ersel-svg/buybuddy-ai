"""
Product ID-based Stratified Data Splitter.

Splits products into train/val/test sets ensuring:
- No data leakage: Same product_id never appears in multiple splits
- Stratified by brand: Each split has similar brand distribution
- Reproducible: Same seed produces same splits

NOTE: We split by product_id (not UPC/barcode) because:
- product_id is the unique canonical identifier
- A product can have multiple barcodes (UPC, EAN, short_code)
- The model learns to recognize products, not barcodes
"""

import random
from collections import defaultdict
from typing import Optional

from supabase import create_client, Client


class ProductSplitter:
    """
    Split products into train/val/test sets by product_id.

    Ensures no product appears in multiple splits (no data leakage).
    Stratifies by brand to maintain distribution across splits.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        random_seed: int = 42,
    ):
        """
        Initialize the splitter.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
            random_seed: Random seed for reproducibility
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.random_seed = random_seed
        # Use Supabase SDK instead of raw httpx for better DNS handling
        self.client: Client = create_client(supabase_url, supabase_key)

    def fetch_products(
        self,
        product_ids: Optional[list[str]] = None,
        min_frames: int = 1,
    ) -> list[dict]:
        """
        Fetch products from Supabase.

        Args:
            product_ids: Optional list of specific product IDs to fetch
            min_frames: Minimum number of frames required

        Returns:
            List of product dictionaries
        """
        # Build query using Supabase SDK
        query = self.client.table("products").select(
            "id,barcode,brand_name,frames_path,frame_count,identifiers"
        )

        # Filter by frame count
        query = query.gte("frame_count", min_frames)

        # Filter by specific IDs if provided
        if product_ids:
            query = query.in_("id", product_ids)

        # Execute query
        response = query.execute()
        products = response.data

        print(f"Fetched {len(products)} products from database")

        return products

    def split(
        self,
        products: Optional[list[dict]] = None,
        product_ids: Optional[list[str]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_frames: int = 1,
        stratify_by_brand: bool = True,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Split products into train/val/test sets.

        Args:
            products: Pre-fetched products (if None, fetches from DB)
            product_ids: Specific product IDs to include
            train_ratio: Fraction for training (default 0.70)
            val_ratio: Fraction for validation (default 0.15)
            test_ratio: Fraction for testing (default 0.15)
            min_frames: Minimum frames per product
            stratify_by_brand: Whether to stratify by brand

        Returns:
            Tuple of (train_products, val_products, test_products)
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        # Fetch products if not provided
        if products is None:
            products = self.fetch_products(
                product_ids=product_ids,
                min_frames=min_frames,
            )

        if len(products) < 3:
            raise ValueError(f"Need at least 3 products, got {len(products)}")

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        if stratify_by_brand:
            return self._stratified_split(products, train_ratio, val_ratio, test_ratio)
        else:
            return self._random_split(products, train_ratio, val_ratio, test_ratio)

    def _random_split(
        self,
        products: list[dict],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Simple random split."""
        shuffled = products.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        return train, val, test

    def _stratified_split(
        self,
        products: list[dict],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Stratified split by brand.

        Ensures each split has similar brand distribution.
        """
        # Group products by brand
        brand_groups = defaultdict(list)
        for product in products:
            brand = product.get("brand_name") or "unknown"
            brand_groups[brand].append(product)

        train, val, test = [], [], []

        # Collect single-product brands separately for random assignment
        single_product_brands = []

        # Split each brand group proportionally
        for brand, brand_products in brand_groups.items():
            random.shuffle(brand_products)
            n = len(brand_products)

            # For single-product brands, collect for later random assignment
            if n == 1:
                single_product_brands.append(brand_products[0])
                continue

            # For brands with 2 products: 1 train, 1 test (skip val)
            if n == 2:
                train.append(brand_products[0])
                test.append(brand_products[1])
                continue

            # For brands with 3+ products: normal proportional split
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            # Ensure at least 1 in each split
            train_end = max(train_end, 1)
            val_end = max(val_end, train_end + 1)

            train.extend(brand_products[:train_end])
            val.extend(brand_products[train_end:val_end])
            test.extend(brand_products[val_end:])

        # Randomly assign single-product brands according to ratios
        # This prevents all single-product brands from going to test
        random.shuffle(single_product_brands)
        n_single = len(single_product_brands)
        if n_single > 0:
            train_end = int(n_single * train_ratio)
            val_end = train_end + int(n_single * val_ratio)

            # Ensure at least some go to train if we have enough
            if n_single >= 3:
                train_end = max(train_end, 1)
                val_end = max(val_end, train_end + 1)
            elif n_single >= 1:
                # With 1-2 single products, put them in train
                train_end = max(train_end, min(n_single, 1))
                val_end = train_end

            train.extend(single_product_brands[:train_end])
            val.extend(single_product_brands[train_end:val_end])
            test.extend(single_product_brands[val_end:])

        # Final shuffle within each split
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        # Print statistics
        self._print_split_stats(train, val, test)

        return train, val, test

    def _print_split_stats(
        self,
        train: list[dict],
        val: list[dict],
        test: list[dict],
    ):
        """Print split statistics."""
        total = len(train) + len(val) + len(test)

        print(f"\nData Split Statistics:")
        print(f"  Total products: {total}")
        print(f"  Train: {len(train)} ({len(train)/total:.1%})")
        print(f"  Val: {len(val)} ({len(val)/total:.1%})")
        print(f"  Test: {len(test)} ({len(test)/total:.1%})")

        # Brand distribution
        def count_brands(products):
            brands = set(p.get("brand_name") or "unknown" for p in products)
            return len(brands)

        print(f"\nBrand distribution:")
        print(f"  Train brands: {count_brands(train)}")
        print(f"  Val brands: {count_brands(val)}")
        print(f"  Test brands: {count_brands(test)}")

        # Frame count statistics
        def frame_stats(products):
            frames = [p.get("frame_count", 0) for p in products]
            return sum(frames), min(frames) if frames else 0, max(frames) if frames else 0

        for name, products in [("Train", train), ("Val", val), ("Test", test)]:
            total_frames, min_f, max_f = frame_stats(products)
            print(f"  {name} frames: {total_frames} total, {min_f}-{max_f} range")


class ProductIDSplitter(ProductSplitter):
    """
    Alias for ProductSplitter (product_id based splitting).

    Kept for backward compatibility and clarity.
    """
    pass


# Legacy alias - UPC-based splitting is deprecated
# Use ProductSplitter (product_id based) instead
UPCStratifiedSplitter = ProductSplitter


def create_identifier_mapping(products: list[dict]) -> dict:
    """
    Create a mapping from product_id to all its identifiers.

    This is used after training to look up what identifiers
    a product_id maps to.

    Args:
        products: List of product dictionaries

    Returns:
        Dictionary mapping product_id to identifier info:
        {
            "product_id_1": {
                "barcode": "123456789",
                "short_code": "ABC123",
                "upc": "123456789012",
                "brand_name": "Brand X"
            },
            ...
        }
    """
    mapping = {}

    for product in products:
        product_id = product.get("id")
        if not product_id:
            continue

        # Extract identifiers from JSON field if available
        identifiers = product.get("identifiers") or {}
        mapping[product_id] = {
            "barcode": product.get("barcode"),
            "short_code": identifiers.get("short_code"),
            "upc": identifiers.get("upc"),
            "brand_name": product.get("brand_name"),
        }

    return mapping


def save_identifier_mapping(mapping: dict, path: str):
    """Save identifier mapping to JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved identifier mapping to {path}")


def load_identifier_mapping(path: str) -> dict:
    """Load identifier mapping from JSON file."""
    import json
    with open(path, "r") as f:
        return json.load(f)
