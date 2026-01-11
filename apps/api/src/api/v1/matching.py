"""Matching API router for product matching with FAISS."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class MatchCandidate(BaseModel):
    """Match candidate schema."""

    id: str
    image_path: str
    image_url: str
    similarity: float
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    """Search request schema."""

    top_k: int = 10


class SearchResponse(BaseModel):
    """Search response schema."""

    candidates: list[MatchCandidate]


class ApproveMatchRequest(BaseModel):
    """Request to approve/reject a match."""

    match_id: str
    is_approved: bool


class ProductMatch(BaseModel):
    """Product match record schema."""

    id: str
    reference_upc: str
    candidate_path: str
    similarity: float
    is_approved: Optional[bool] = None
    created_at: datetime


# ===========================================
# Mock Data
# ===========================================

MOCK_UPCS = ["0012345678901", "0012345678902", "0012345678903"]

MOCK_CANDIDATES: dict[str, list[dict]] = {
    "0012345678901": [
        {
            "id": str(uuid4()),
            "image_path": "/real/coca-cola/image1.jpg",
            "image_url": "https://storage.example.com/real/coca-cola/image1.jpg",
            "similarity": 0.95,
            "metadata": {"source": "retail_dataset"},
        },
        {
            "id": str(uuid4()),
            "image_path": "/real/coca-cola/image2.jpg",
            "image_url": "https://storage.example.com/real/coca-cola/image2.jpg",
            "similarity": 0.89,
            "metadata": {"source": "retail_dataset"},
        },
    ]
}

MOCK_MATCHES: list[dict] = []


# ===========================================
# Endpoints
# ===========================================


@router.get("/upcs")
async def list_upcs() -> list[str]:
    """List all UPCs available for matching."""
    return MOCK_UPCS


@router.post("/upcs/{upc}/search", response_model=SearchResponse)
async def search_matches(upc: str, request: SearchRequest) -> SearchResponse:
    """Search for matching candidates for a UPC."""
    if upc not in MOCK_UPCS:
        raise HTTPException(status_code=404, detail="UPC not found")

    candidates = MOCK_CANDIDATES.get(upc, [])

    # Sort by similarity and limit
    sorted_candidates = sorted(candidates, key=lambda x: x["similarity"], reverse=True)
    limited = sorted_candidates[: request.top_k]

    return SearchResponse(candidates=[MatchCandidate(**c) for c in limited])


@router.post("/approve")
async def approve_match(request: ApproveMatchRequest) -> dict[str, str]:
    """Approve or reject a match."""
    # TODO: Save to database
    match_record = {
        "id": request.match_id,
        "is_approved": request.is_approved,
        "updated_at": datetime.now().isoformat(),
    }

    return {"status": "approved" if request.is_approved else "rejected"}


@router.get("/matches")
async def list_matches(
    upc: Optional[str] = Query(None, description="Filter by UPC"),
    is_approved: Optional[bool] = Query(None, description="Filter by approval status"),
) -> list[ProductMatch]:
    """List all product matches."""
    matches = MOCK_MATCHES.copy()

    if upc:
        matches = [m for m in matches if m["reference_upc"] == upc]
    if is_approved is not None:
        matches = [m for m in matches if m.get("is_approved") == is_approved]

    return [ProductMatch(**m) for m in matches]
