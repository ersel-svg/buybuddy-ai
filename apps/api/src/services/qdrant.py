"""Qdrant vector database client for embedding storage and similarity search."""

from typing import Optional, Any
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS
import re

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    ScrollRequest,
    SearchRequest,
    PayloadSchemaType,
)

from config import settings


class QdrantService:
    """Service class for Qdrant vector database operations."""

    def __init__(self) -> None:
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Lazy load Qdrant client."""
        if self._client is None:
            if not settings.qdrant_url:
                raise ValueError("QDRANT_URL not configured")

            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
                timeout=60,
            )
        return self._client

    def is_configured(self) -> bool:
        """Check if Qdrant is configured."""
        return bool(settings.qdrant_url)

    # =========================================
    # Collection Management
    # =========================================

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: str = "Cosine",
    ) -> bool:
        """
        Create a new collection for embeddings.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (768 for DINOv2-base)
            distance: Distance metric (Cosine, Euclid, Dot)

        Returns:
            True if created, False if already exists
        """
        # Check if collection exists
        collections = self.client.get_collections()
        existing = [c.name for c in collections.collections]

        if collection_name in existing:
            print(f"[Qdrant] Collection '{collection_name}' already exists")
            return False

        # Map distance string to enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(distance, Distance.COSINE),
            ),
        )

        print(f"[Qdrant] Created collection '{collection_name}' (dim={vector_size})")

        # Create default payload indexes for filtering
        await self._create_default_indexes(collection_name)

        return True

    async def _create_default_indexes(self, collection_name: str) -> None:
        """Create default payload indexes required for filtering operations."""
        default_indexes = [
            ("source", PayloadSchemaType.KEYWORD),
            ("is_primary", PayloadSchemaType.BOOL),
            ("product_id", PayloadSchemaType.KEYWORD),
            ("barcode", PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in default_indexes:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                print(f"[Qdrant] Created index '{field_name}' on '{collection_name}'")
            except Exception as e:
                # Index might already exist
                print(f"[Qdrant] Index '{field_name}' may already exist: {e}")

    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_type: str = "keyword",
    ) -> bool:
        """
        Create a payload index for efficient filtering.

        Args:
            collection_name: Target collection
            field_name: Field to index
            field_type: Type of index (keyword, integer, float, bool, geo, text)

        Returns:
            True if created successfully
        """
        type_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "bool": PayloadSchemaType.BOOL,
            "geo": PayloadSchemaType.GEO,
            "text": PayloadSchemaType.TEXT,
        }

        schema_type = type_map.get(field_type.lower(), PayloadSchemaType.KEYWORD)

        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            print(f"[Qdrant] Created {field_type} index for '{field_name}' on '{collection_name}'")
            return True
        except Exception as e:
            print(f"[Qdrant] Error creating index: {e}")
            raise

    async def ensure_indexes(self, collection_name: str) -> dict[str, Any]:
        """
        Ensure all required indexes exist on a collection.
        Creates missing indexes without affecting existing ones.

        Args:
            collection_name: Target collection

        Returns:
            Dict with created and existing indexes
        """
        required_indexes = [
            ("source", "keyword"),
            ("is_primary", "bool"),
            ("product_id", "keyword"),
            ("barcode", "keyword"),
        ]

        results = {"created": [], "existing": [], "failed": []}

        for field_name, field_type in required_indexes:
            try:
                await self.create_payload_index(collection_name, field_name, field_type)
                results["created"].append(field_name)
            except Exception as e:
                error_msg = str(e).lower()
                if "already exists" in error_msg or "conflict" in error_msg:
                    results["existing"].append(field_name)
                else:
                    results["failed"].append({"field": field_name, "error": str(e)})

        return results

    async def get_collection_indexes(self, collection_name: str) -> list[dict[str, Any]]:
        """
        Get list of payload indexes on a collection.

        Args:
            collection_name: Target collection

        Returns:
            List of index info dicts
        """
        try:
            info = self.client.get_collection(collection_name=collection_name)
            indexes = []

            if info.payload_schema:
                for field_name, field_info in info.payload_schema.items():
                    indexes.append({
                        "field_name": field_name,
                        "data_type": str(field_info.data_type) if field_info.data_type else "unknown",
                        "points": field_info.points if hasattr(field_info, 'points') else 0,
                    })

            return indexes
        except Exception as e:
            print(f"[Qdrant] Error getting indexes: {e}")
            return []

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"[Qdrant] Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"[Qdrant] Error deleting collection: {e}")
            return False

    async def get_collection_info(self, collection_name: str) -> Optional[dict[str, Any]]:
        """Get collection info including vector count."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            # Handle different qdrant-client versions
            vectors_count = getattr(info, 'vectors_count', None) or info.points_count or 0
            points_count = info.points_count or 0
            status = info.status.value if hasattr(info.status, 'value') else str(info.status)

            # Get vector size from config
            vector_size = 0
            if hasattr(info.config, 'params') and info.config.params:
                vec_cfg = info.config.params.vectors
                if hasattr(vec_cfg, 'size'):
                    vector_size = vec_cfg.size
                elif isinstance(vec_cfg, dict):
                    # Named vectors
                    first_vec = next(iter(vec_cfg.values()), None)
                    if first_vec and hasattr(first_vec, 'size'):
                        vector_size = first_vec.size

            return {
                "name": collection_name,
                "vectors_count": vectors_count,
                "points_count": points_count,
                "status": status,
                "vector_size": vector_size,
            }
        except Exception as e:
            print(f"[Qdrant] Error getting collection info: {e}")
            return None

    async def list_collections(self) -> list[str]:
        """List all collection names."""
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    async def list_collections_with_info(self) -> list[dict[str, Any]]:
        """List all collections with detailed info."""
        collections = self.client.get_collections()
        result = []
        for c in collections.collections:
            try:
                info = self.client.get_collection(collection_name=c.name)
                # Extract model name from collection name (e.g., products_dinov2_base -> dinov2_base)
                model_name = None
                if "_" in c.name:
                    parts = c.name.split("_", 1)
                    if parts[0] in ["products", "cutouts"]:
                        model_name = parts[1]

                result.append({
                    "name": c.name,
                    "vectors_count": info.vectors_count or 0,
                    "points_count": info.points_count or 0,
                    "status": info.status.value if info.status else "unknown",
                    "vector_size": info.config.params.vectors.size if info.config and info.config.params and info.config.params.vectors else 0,
                    "model_name": model_name,
                })
            except Exception as e:
                print(f"[Qdrant] Error getting info for collection {c.name}: {e}")
                result.append({
                    "name": c.name,
                    "vectors_count": 0,
                    "points_count": 0,
                    "status": "error",
                    "vector_size": 0,
                    "model_name": None,
                })
        return result

    async def get_product_embeddings(
        self,
        collection_name: str,
        product_id: str,
        with_vectors: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get all embeddings for a product (multi-view support).

        Searches for points where payload.product_id matches the given product_id.
        Returns all frames (e.g., product_id_0, product_id_1, etc.).

        Args:
            collection_name: Target collection
            product_id: Product ID to search for
            with_vectors: Include vectors in results

        Returns:
            List of points with id, payload, and optionally vector
        """
        points, _ = await self.scroll(
            collection_name=collection_name,
            filter_conditions={"product_id": product_id},
            limit=100,  # Max frames per product
            with_vectors=with_vectors,
        )
        return points

    # =========================================
    # Point Operations
    # =========================================

    def _normalize_point_id(self, point_id: str) -> str:
        """
        Normalize point ID to a valid Qdrant format (UUID or unsigned integer).

        Qdrant only accepts:
        - Valid UUID strings (e.g., "550e8400-e29b-41d4-a716-446655440000")
        - Unsigned integers

        For compound IDs like "uuid_frame_index", we generate a deterministic
        UUID using uuid5 so the same input always produces the same output.

        Args:
            point_id: Original point ID string

        Returns:
            Valid Qdrant point ID (UUID string)
        """
        if not point_id:
            return str(uuid4())

        # Check if it's already a valid UUID
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        if uuid_pattern.match(point_id):
            return point_id

        # Check if it's a valid unsigned integer
        if point_id.isdigit():
            return point_id

        # For compound IDs (e.g., "uuid_0", "uuid_frame_5"), generate deterministic UUID
        # Using uuid5 ensures same input always produces same UUID
        return str(uuid5(NAMESPACE_DNS, f"buybuddy.embedding.{point_id}"))

    async def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> int:
        """
        Upsert points (vectors with metadata) to collection.

        Args:
            collection_name: Target collection
            points: List of dicts with:
                - id: UUID, unsigned int, or compound string (will be normalized)
                - vector: List of floats
                - payload: Dict of metadata

        Returns:
            Number of points upserted
        """
        if not points:
            return 0

        qdrant_points = []
        for p in points:
            original_id = p.get("id") or str(uuid4())
            # Normalize ID to valid Qdrant format
            point_id = self._normalize_point_id(original_id)

            # Store original_id in payload for reference if it was transformed
            payload = p.get("payload", {}).copy()
            if point_id != original_id:
                payload["original_point_id"] = original_id

            qdrant_points.append(
                PointStruct(
                    id=point_id,
                    vector=p["vector"],
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points,
            wait=True,
        )

        return len(qdrant_points)

    async def delete_points(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> int:
        """Delete points by IDs."""
        if not point_ids:
            return 0

        self.client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.PointIdsList(points=point_ids),
            wait=True,
        )

        return len(point_ids)

    async def delete_points_by_filter(
        self,
        collection_name: str,
        filter_conditions: dict[str, Any],
    ) -> bool:
        """
        Delete points matching filter conditions.

        Args:
            collection_name: Target collection
            filter_conditions: Dict of field -> value conditions

        Example:
            delete_points_by_filter("embeddings", {"product_id": "uuid-123"})
        """
        must_conditions = []
        for field, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append(
                    FieldCondition(key=field, match=MatchAny(any=value))
                )
            else:
                must_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )

        self.client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=Filter(must=must_conditions)
            ),
            wait=True,
        )

        return True

    # =========================================
    # Search Operations
    # =========================================

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 100,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict[str, Any]] = None,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            collection_name: Target collection
            query_vector: Query embedding vector
            limit: Max results to return
            score_threshold: Min similarity score (0-1 for cosine)
            filter_conditions: Optional filter dict
            with_payload: Include metadata in results

        Returns:
            List of results with id, score, and payload
        """
        # Build filter
        search_filter = None
        if filter_conditions:
            must_conditions = []
            should_conditions = []

            for field, value in filter_conditions.items():
                if field.startswith("should_"):
                    actual_field = field[7:]  # Remove "should_" prefix
                    if isinstance(value, list):
                        should_conditions.append(
                            FieldCondition(key=actual_field, match=MatchAny(any=value))
                        )
                    else:
                        should_conditions.append(
                            FieldCondition(key=actual_field, match=MatchValue(value=value))
                        )
                else:
                    if isinstance(value, list):
                        must_conditions.append(
                            FieldCondition(key=field, match=MatchAny(any=value))
                        )
                    else:
                        must_conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )

            search_filter = Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None,
            )

        # Use query_points (qdrant-client 1.9+)
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
            with_payload=with_payload,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload if with_payload else None,
            }
            for r in response.points
        ]

    async def search_by_id(
        self,
        collection_name: str,
        point_id: str,
        limit: int = 100,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search using an existing point's vector.

        Args:
            collection_name: Target collection
            point_id: ID of point to use as query
            limit: Max results
            score_threshold: Min similarity
            filter_conditions: Optional filter

        Returns:
            List of similar points (excluding the query point)
        """
        # Get the point's vector
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=True,
        )

        if not points:
            return []

        query_vector = points[0].vector

        # Search and exclude self
        results = await self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit + 1,  # +1 to account for self
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

        # Remove self from results
        return [r for r in results if r["id"] != point_id][:limit]

    # =========================================
    # Scroll (for export)
    # =========================================

    async def scroll(
        self,
        collection_name: str,
        filter_conditions: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: Optional[str] = None,
        with_vectors: bool = False,
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """
        Scroll through collection (for export/iteration).

        Args:
            collection_name: Target collection
            filter_conditions: Optional filter
            limit: Batch size
            offset: Offset point ID for pagination
            with_vectors: Include vectors in results

        Returns:
            Tuple of (points, next_offset)
        """
        # Build filter
        scroll_filter = None
        if filter_conditions:
            must_conditions = []
            for field, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
            scroll_filter = Filter(must=must_conditions)

        result = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )

        points = [
            {
                "id": str(p.id),
                "payload": p.payload,
                "vector": p.vector if with_vectors else None,
            }
            for p in result[0]
        ]

        next_offset = result[1]

        return points, next_offset

    async def scroll_all(
        self,
        collection_name: str,
        filter_conditions: Optional[dict[str, Any]] = None,
        with_vectors: bool = False,
        batch_size: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Scroll through entire collection (use carefully for large collections).

        Args:
            collection_name: Target collection
            filter_conditions: Optional filter
            with_vectors: Include vectors
            batch_size: Batch size for scrolling

        Returns:
            All matching points
        """
        all_points = []
        offset = None

        while True:
            points, next_offset = await self.scroll(
                collection_name=collection_name,
                filter_conditions=filter_conditions,
                limit=batch_size,
                offset=offset,
                with_vectors=with_vectors,
            )

            all_points.extend(points)

            if next_offset is None or len(points) < batch_size:
                break

            offset = next_offset

        return all_points

    # =========================================
    # Utility Methods
    # =========================================

    async def get_point(
        self,
        collection_name: str,
        point_id: str,
        with_vector: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Get a single point by ID."""
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=with_vector,
            )
            if points:
                return {
                    "id": str(points[0].id),
                    "payload": points[0].payload,
                    "vector": points[0].vector if with_vector else None,
                }
            return None
        except Exception:
            return None

    async def count_points(
        self,
        collection_name: str,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> int:
        """Count points in collection with optional filter."""
        count_filter = None
        if filter_conditions:
            must_conditions = []
            for field, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
            count_filter = Filter(must=must_conditions)

        result = self.client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=True,
        )

        return result.count

    # =========================================
    # Snapshot Operations (for export)
    # =========================================

    async def create_snapshot(self, collection_name: str) -> dict[str, Any]:
        """Create a snapshot of a collection."""
        try:
            result = self.client.create_snapshot(collection_name=collection_name)
            return {
                "name": result.name,
                "creation_time": str(result.creation_time) if result.creation_time else None,
                "size": result.size,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to create snapshot: {e}")

    async def get_snapshot_url(
        self,
        collection_name: str,
        snapshot_name: str,
    ) -> str:
        """Get download URL for a snapshot."""
        # Build the snapshot download URL
        base_url = settings.qdrant_url.rstrip("/")
        url = f"{base_url}/collections/{collection_name}/snapshots/{snapshot_name}"
        return url

    async def list_snapshots(self, collection_name: str) -> list[dict[str, Any]]:
        """List all snapshots for a collection."""
        try:
            snapshots = self.client.list_snapshots(collection_name=collection_name)
            return [
                {
                    "name": s.name,
                    "creation_time": str(s.creation_time) if s.creation_time else None,
                    "size": s.size,
                }
                for s in snapshots
            ]
        except Exception as e:
            print(f"[Qdrant] Error listing snapshots: {e}")
            return []

    async def delete_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
    ) -> bool:
        """Delete a snapshot."""
        try:
            self.client.delete_snapshot(
                collection_name=collection_name,
                snapshot_name=snapshot_name,
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Error deleting snapshot: {e}")
            return False


# Singleton instance
qdrant_service = QdrantService()
