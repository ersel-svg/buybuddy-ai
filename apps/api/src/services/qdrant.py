"""Qdrant vector database client for embedding storage and similarity search."""

from typing import Optional, Any
from uuid import UUID, uuid4

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
        return True

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
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "vector_size": info.config.params.vectors.size,
            }
        except Exception as e:
            print(f"[Qdrant] Error getting collection info: {e}")
            return None

    async def list_collections(self) -> list[str]:
        """List all collection names."""
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    # =========================================
    # Point Operations
    # =========================================

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
                - id: UUID or string (optional, auto-generated if not provided)
                - vector: List of floats
                - payload: Dict of metadata

        Returns:
            Number of points upserted
        """
        if not points:
            return 0

        qdrant_points = []
        for p in points:
            point_id = p.get("id") or str(uuid4())
            qdrant_points.append(
                PointStruct(
                    id=point_id,
                    vector=p["vector"],
                    payload=p.get("payload", {}),
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

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
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
            for r in results
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
