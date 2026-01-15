"""
Export Service for embedding vectors.

Supports multiple formats:
- JSON: Human-readable, includes metadata
- NumPy: .npz file with vectors and IDs
- FAISS: Binary index file for fast similarity search
- Qdrant Snapshot: Native Qdrant backup format
"""

import json
import io
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime

import numpy as np

from services.qdrant import qdrant_service
from services.supabase import supabase_service


ExportFormat = Literal["json", "numpy", "faiss", "qdrant_snapshot"]


class ExportService:
    """Service for exporting embeddings to various formats."""

    def __init__(self):
        self.bucket_name = "exports"

    async def export_embeddings(
        self,
        model_id: str,
        collection_name: str,
        format: ExportFormat,
        export_id: str,
    ) -> dict:
        """
        Export embeddings from Qdrant to specified format.

        Args:
            model_id: Embedding model ID
            collection_name: Qdrant collection name
            format: Export format (json, numpy, faiss, qdrant_snapshot)
            export_id: Export record ID for tracking

        Returns:
            Dict with file_url, vector_count, file_size_bytes
        """
        # For snapshot, we don't need to fetch vectors
        if format == "qdrant_snapshot":
            result = await self._export_qdrant_snapshot(collection_name, export_id)
            # Get vector count from collection info
            info = await qdrant_service.get_collection_info(collection_name)
            result["vector_count"] = info.get("points_count", 0) if info else 0
            return result

        # Fetch all vectors from Qdrant (with vectors for other formats)
        print(f"Fetching vectors from collection: {collection_name}")
        points = await qdrant_service.scroll_all(collection_name, with_vectors=True)

        if not points:
            return {
                "file_url": None,
                "vector_count": 0,
                "file_size_bytes": 0,
            }

        print(f"Retrieved {len(points)} vectors")

        # Generate export based on format
        if format == "json":
            result = await self._export_json(points, export_id)
        elif format == "numpy":
            result = await self._export_numpy(points, export_id)
        elif format == "faiss":
            result = await self._export_faiss(points, export_id)
        else:
            raise ValueError(f"Unknown export format: {format}")

        result["vector_count"] = len(points)
        return result

    async def _export_json(self, points: list[dict], export_id: str) -> dict:
        """Export to JSON format. Points are dicts with id, vector, payload keys."""
        data = {
            "export_id": export_id,
            "created_at": datetime.utcnow().isoformat(),
            "vector_count": len(points),
            "embeddings": []
        }

        for point in points:
            data["embeddings"].append({
                "id": str(point["id"]),
                "vector": point["vector"],
                "payload": point.get("payload", {}),
            })

        # Serialize to JSON
        json_str = json.dumps(data, indent=2)
        json_bytes = json_str.encode("utf-8")

        # Upload to storage
        filename = f"embeddings_{export_id}.json"
        file_url = await self._upload_to_storage(json_bytes, filename, "application/json")

        return {
            "file_url": file_url,
            "file_size_bytes": len(json_bytes),
        }

    async def _export_numpy(self, points: list[dict], export_id: str) -> dict:
        """Export to NumPy .npz format. Points are dicts with id, vector, payload keys."""
        # Extract vectors and IDs
        vectors = np.array([point["vector"] for point in points], dtype=np.float32)
        ids = np.array([str(point["id"]) for point in points])

        # Extract metadata as JSON strings
        payloads = np.array([json.dumps(point.get("payload", {})) for point in points])

        # Save to npz
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            vectors=vectors,
            ids=ids,
            payloads=payloads,
        )
        buffer.seek(0)
        npz_bytes = buffer.read()

        # Upload to storage
        filename = f"embeddings_{export_id}.npz"
        file_url = await self._upload_to_storage(npz_bytes, filename, "application/octet-stream")

        return {
            "file_url": file_url,
            "file_size_bytes": len(npz_bytes),
        }

    async def _export_faiss(self, points: list[dict], export_id: str) -> dict:
        """Export to FAISS index format. Points are dicts with id, vector, payload keys."""
        try:
            import faiss
        except ImportError:
            raise RuntimeError("FAISS not installed. Install with: pip install faiss-cpu")

        # Extract vectors
        vectors = np.array([point["vector"] for point in points], dtype=np.float32)
        dim = vectors.shape[1]

        # Create FAISS index (using inner product for cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        # Also save ID mapping
        ids = [str(point["id"]) for point in points]
        id_mapping = {i: id_str for i, id_str in enumerate(ids)}

        # Save index to bytes
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            faiss.write_index(index, f.name)
            f.seek(0)
            index_path = f.name

        with open(index_path, "rb") as f:
            index_bytes = f.read()

        Path(index_path).unlink()  # Cleanup temp file

        # Upload index
        index_filename = f"embeddings_{export_id}.faiss"
        index_url = await self._upload_to_storage(index_bytes, index_filename, "application/octet-stream")

        # Also upload ID mapping
        mapping_bytes = json.dumps(id_mapping).encode("utf-8")
        mapping_filename = f"embeddings_{export_id}_ids.json"
        await self._upload_to_storage(mapping_bytes, mapping_filename, "application/json")

        return {
            "file_url": index_url,
            "file_size_bytes": len(index_bytes),
        }

    async def _export_qdrant_snapshot(self, collection_name: str, export_id: str) -> dict:
        """Export Qdrant collection snapshot."""
        if not qdrant_service.is_configured():
            raise RuntimeError("Qdrant not configured")

        # Create snapshot via Qdrant API
        try:
            snapshot_info = await qdrant_service.create_snapshot(collection_name)
            snapshot_name = snapshot_info.get("name")

            if not snapshot_name:
                raise RuntimeError("Failed to create snapshot")

            # Get snapshot download URL
            snapshot_url = await qdrant_service.get_snapshot_url(collection_name, snapshot_name)

            return {
                "file_url": snapshot_url,
                "file_size_bytes": snapshot_info.get("size", 0),
            }
        except Exception as e:
            raise RuntimeError(f"Snapshot export failed: {e}")

    async def _upload_to_storage(
        self,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        """Upload bytes to Supabase Storage."""
        try:
            # Upload to exports bucket
            path = f"exports/{filename}"
            result = supabase_service.client.storage.from_(self.bucket_name).upload(
                path,
                data,
                {"content-type": content_type},
            )

            # Get public URL
            url = supabase_service.client.storage.from_(self.bucket_name).get_public_url(path)
            return url

        except Exception as e:
            print(f"Storage upload error: {e}")
            # Fallback: return None, caller should handle
            return None


# Singleton
export_service = ExportService()
