"""
Triplet Mining API router for finding hard negatives and creating training triplets.

Supports:
- Mining hard triplets from Qdrant embeddings
- Cross-domain triplet mining (synthetic → real)
- Triplet difficulty classification
- Statistics and visualization
"""

from typing import Optional, Literal
from datetime import datetime
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from services.supabase import SupabaseService, supabase_service
from services.qdrant import QdrantService, qdrant_service
from auth.dependencies import get_current_user

# Router with authentication required
router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================

class TripletMiningConfig(BaseModel):
    """Configuration for triplet mining."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    # Source
    dataset_id: Optional[str] = None  # Filter to specific dataset
    embedding_model_id: str  # Required - which model's embeddings to use
    collection_name: str  # Qdrant collection to mine from

    # Mining thresholds
    hard_negative_threshold: float = Field(0.7, ge=0.3, le=0.95)
    positive_threshold: float = Field(0.9, ge=0.7, le=1.0)
    max_triplets_per_anchor: int = Field(10, ge=1, le=100)

    # Domain options
    include_cross_domain: bool = True  # Include synthetic-real pairs


class TripletMiningResponse(BaseModel):
    """Response for triplet mining run creation."""
    id: str
    status: str
    message: str


class TripletResponse(BaseModel):
    """Single triplet response."""
    id: str
    anchor_product_id: str
    positive_product_id: str
    negative_product_id: str
    anchor_frame_idx: int
    positive_frame_idx: int
    negative_frame_idx: int
    anchor_positive_sim: float
    anchor_negative_sim: float
    margin: float
    difficulty: str
    is_cross_domain: bool


class TripletStatsResponse(BaseModel):
    """Statistics for a mining run."""
    total_triplets: int
    hard_count: int
    semi_hard_count: int
    easy_count: int
    cross_domain_count: int
    avg_margin: float
    min_margin: float
    max_margin: float


class MiningRunResponse(BaseModel):
    """Mining run response."""
    id: str
    name: str
    status: str
    collection_name: str
    total_triplets: Optional[int]
    hard_triplets: Optional[int]
    cross_domain_triplets: Optional[int]
    created_at: datetime


# ===========================================
# Dependencies
# ===========================================

def get_supabase() -> SupabaseService:
    return supabase_service


def get_qdrant() -> QdrantService:
    return qdrant_service


# ===========================================
# Endpoints
# ===========================================

@router.post("/mine", response_model=TripletMiningResponse)
async def start_triplet_mining(
    config: TripletMiningConfig,
    background_tasks: BackgroundTasks,
    db: SupabaseService = Depends(get_supabase),
    qdrant: QdrantService = Depends(get_qdrant),
):
    """
    Start a triplet mining job.

    Mines hard triplets from embeddings stored in Qdrant.
    - Hard negatives: Different products with high similarity
    - Cross-domain: Synthetic anchors with real negatives
    """
    # Verify collection exists
    collections = await qdrant.list_collections()
    if config.collection_name not in collections:
        raise HTTPException(400, f"Collection '{config.collection_name}' not found")

    # Create mining run record
    run_data = {
        "name": config.name,
        "description": config.description,
        "dataset_id": config.dataset_id,
        "embedding_model_id": config.embedding_model_id,
        "collection_name": config.collection_name,
        "hard_negative_threshold": config.hard_negative_threshold,
        "positive_threshold": config.positive_threshold,
        "max_triplets_per_anchor": config.max_triplets_per_anchor,
        "include_cross_domain": config.include_cross_domain,
        "status": "pending",
    }

    result = db.client.table("triplet_mining_runs").insert(run_data).execute()
    if not result.data:
        raise HTTPException(500, "Failed to create mining run")

    run_id = result.data[0]["id"]

    # Start mining in background
    background_tasks.add_task(
        mine_triplets_task,
        run_id=run_id,
        config=config,
        db=db,
        qdrant=qdrant,
    )

    return TripletMiningResponse(
        id=run_id,
        status="pending",
        message="Triplet mining started in background"
    )


async def mine_triplets_task(
    run_id: str,
    config: TripletMiningConfig,
    db: SupabaseService,
    qdrant: QdrantService,
):
    """Background task for triplet mining."""
    try:
        # Update status to running
        db.client.table("triplet_mining_runs").update({
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
        }).eq("id", run_id).execute()

        # Get all embeddings from collection
        embeddings_data = await get_all_embeddings(qdrant, config.collection_name)

        if not embeddings_data:
            raise ValueError("No embeddings found in collection")

        # Group by product_id
        product_embeddings = defaultdict(list)
        for emb in embeddings_data:
            product_id = emb["payload"].get("product_id")
            if product_id:
                product_embeddings[product_id].append({
                    "vector": np.array(emb["vector"]),
                    "frame_idx": emb["payload"].get("frame_index", 0),
                    "domain": emb["payload"].get("domain", "unknown"),
                })

        # Mine triplets
        triplets = []
        product_ids = list(product_embeddings.keys())

        for anchor_id in product_ids:
            anchor_data = product_embeddings[anchor_id]

            for anchor_emb in anchor_data:
                anchor_vec = anchor_emb["vector"]
                anchor_frame = anchor_emb["frame_idx"]
                anchor_domain = anchor_emb["domain"]

                # Find positives (same product, different frame)
                positives = [
                    e for e in anchor_data
                    if e["frame_idx"] != anchor_frame
                ]

                if not positives:
                    continue

                # Find hard negatives
                hard_negatives = []

                for neg_id in product_ids:
                    if neg_id == anchor_id:
                        continue

                    for neg_emb in product_embeddings[neg_id]:
                        neg_vec = neg_emb["vector"]

                        # Cosine similarity
                        sim = float(np.dot(anchor_vec, neg_vec) / (
                            np.linalg.norm(anchor_vec) * np.linalg.norm(neg_vec) + 1e-8
                        ))

                        if sim >= config.hard_negative_threshold:
                            hard_negatives.append({
                                "product_id": neg_id,
                                "frame_idx": neg_emb["frame_idx"],
                                "domain": neg_emb["domain"],
                                "similarity": sim,
                            })

                # Sort by similarity (hardest first)
                hard_negatives.sort(key=lambda x: x["similarity"], reverse=True)
                hard_negatives = hard_negatives[:config.max_triplets_per_anchor]

                # Create triplets
                for pos_emb in positives[:3]:  # Max 3 positives per anchor
                    pos_vec = pos_emb["vector"]
                    pos_sim = float(np.dot(anchor_vec, pos_vec) / (
                        np.linalg.norm(anchor_vec) * np.linalg.norm(pos_vec) + 1e-8
                    ))

                    for neg in hard_negatives:
                        # Classify difficulty based on margin
                        margin = pos_sim - neg["similarity"]

                        if margin < 0.1:
                            difficulty = "hard"
                        elif margin < 0.3:
                            difficulty = "semi_hard"
                        else:
                            difficulty = "easy"

                        # Check cross-domain
                        is_cross_domain = (
                            anchor_domain == "synthetic" and
                            neg["domain"] == "real"
                        )

                        # Skip if not including cross-domain and this is cross-domain
                        if not config.include_cross_domain and is_cross_domain:
                            continue

                        triplets.append({
                            "mining_run_id": run_id,
                            "anchor_product_id": anchor_id,
                            "positive_product_id": anchor_id,
                            "negative_product_id": neg["product_id"],
                            "anchor_frame_idx": anchor_frame,
                            "positive_frame_idx": pos_emb["frame_idx"],
                            "negative_frame_idx": neg["frame_idx"],
                            "anchor_positive_sim": pos_sim,
                            "anchor_negative_sim": neg["similarity"],
                            "difficulty": difficulty,
                            "is_cross_domain": is_cross_domain,
                            "anchor_domain": anchor_domain,
                            "positive_domain": pos_emb["domain"],
                            "negative_domain": neg["domain"],
                        })

        # Batch insert triplets
        if triplets:
            batch_size = 1000
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                db.client.table("mined_triplets").insert(batch).execute()

        # Calculate statistics
        hard_count = sum(1 for t in triplets if t["difficulty"] == "hard")
        semi_hard_count = sum(1 for t in triplets if t["difficulty"] == "semi_hard")
        easy_count = sum(1 for t in triplets if t["difficulty"] == "easy")
        cross_domain_count = sum(1 for t in triplets if t["is_cross_domain"])

        # Update run with stats
        db.client.table("triplet_mining_runs").update({
            "status": "completed",
            "total_anchors": len(product_ids),
            "total_triplets": len(triplets),
            "hard_triplets": hard_count,
            "semi_hard_triplets": semi_hard_count,
            "easy_triplets": easy_count,
            "cross_domain_triplets": cross_domain_count,
            "completed_at": datetime.utcnow().isoformat(),
        }).eq("id", run_id).execute()

        print(f"Triplet mining completed: {len(triplets)} triplets mined")

    except Exception as e:
        print(f"Triplet mining failed: {e}")
        db.client.table("triplet_mining_runs").update({
            "status": "failed",
            "error_message": str(e),
        }).eq("id", run_id).execute()


async def get_all_embeddings(qdrant: QdrantService, collection_name: str) -> list[dict]:
    """Get all embeddings from a Qdrant collection."""
    embeddings = []
    offset = None
    limit = 1000

    while True:
        result = qdrant.client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        points, next_offset = result

        for point in points:
            embeddings.append({
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload or {},
            })

        if next_offset is None or len(points) < limit:
            break

        offset = next_offset

    return embeddings


@router.get("/runs", response_model=list[MiningRunResponse])
async def list_mining_runs(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db: SupabaseService = Depends(get_supabase),
):
    """List triplet mining runs."""
    query = db.client.table("triplet_mining_runs").select("*")

    if status:
        query = query.eq("status", status)

    query = query.order("created_at", desc=True).limit(limit)
    result = query.execute()

    return [
        MiningRunResponse(
            id=r["id"],
            name=r["name"],
            status=r["status"],
            collection_name=r["collection_name"],
            total_triplets=r.get("total_triplets"),
            hard_triplets=r.get("hard_triplets"),
            cross_domain_triplets=r.get("cross_domain_triplets"),
            created_at=r["created_at"],
        )
        for r in result.data
    ]


@router.get("/runs/{run_id}")
async def get_mining_run(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get details of a mining run."""
    result = db.client.table("triplet_mining_runs").select("*").eq("id", run_id).single().execute()

    if not result.data:
        raise HTTPException(404, "Mining run not found")

    return result.data


@router.get("/runs/{run_id}/triplets", response_model=list[TripletResponse])
async def get_triplets(
    run_id: str,
    difficulty: Optional[Literal["hard", "semi_hard", "easy"]] = None,
    cross_domain_only: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: SupabaseService = Depends(get_supabase),
):
    """Get triplets from a mining run."""
    query = db.client.table("mined_triplets").select("*").eq("mining_run_id", run_id)

    if difficulty:
        query = query.eq("difficulty", difficulty)

    if cross_domain_only:
        query = query.eq("is_cross_domain", True)

    query = query.range(offset, offset + limit - 1)
    result = query.execute()

    return [
        TripletResponse(
            id=t["id"],
            anchor_product_id=t["anchor_product_id"],
            positive_product_id=t["positive_product_id"],
            negative_product_id=t["negative_product_id"],
            anchor_frame_idx=t["anchor_frame_idx"],
            positive_frame_idx=t["positive_frame_idx"],
            negative_frame_idx=t["negative_frame_idx"],
            anchor_positive_sim=t["anchor_positive_sim"],
            anchor_negative_sim=t["anchor_negative_sim"],
            margin=t.get("margin", t["anchor_positive_sim"] - t["anchor_negative_sim"]),
            difficulty=t["difficulty"],
            is_cross_domain=t["is_cross_domain"],
        )
        for t in result.data
    ]


@router.get("/runs/{run_id}/stats", response_model=TripletStatsResponse)
async def get_triplet_stats(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get statistics for a mining run."""
    # Use the database function
    result = db.client.rpc("get_triplet_mining_stats", {"run_id": run_id}).execute()

    if not result.data:
        raise HTTPException(404, "Stats not found")

    stats = result.data[0] if isinstance(result.data, list) else result.data

    return TripletStatsResponse(
        total_triplets=stats.get("total_triplets", 0),
        hard_count=stats.get("hard_count", 0),
        semi_hard_count=stats.get("semi_hard_count", 0),
        easy_count=stats.get("easy_count", 0),
        cross_domain_count=stats.get("cross_domain_count", 0),
        avg_margin=stats.get("avg_margin", 0.0),
        min_margin=stats.get("min_margin", 0.0),
        max_margin=stats.get("max_margin", 0.0),
    )


@router.delete("/runs/{run_id}")
async def delete_mining_run(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a mining run and its triplets."""
    # Triplets are deleted via CASCADE
    result = db.client.table("triplet_mining_runs").delete().eq("id", run_id).execute()

    if not result.data:
        raise HTTPException(404, "Mining run not found")

    return {"status": "deleted", "id": run_id}


@router.post("/runs/{run_id}/export")
async def export_triplets(
    run_id: str,
    format: Literal["json", "csv"] = "json",
    difficulty: Optional[Literal["hard", "semi_hard", "easy"]] = None,
    db: SupabaseService = Depends(get_supabase),
):
    """Export triplets for training."""
    query = db.client.table("mined_triplets").select("*").eq("mining_run_id", run_id)

    if difficulty:
        query = query.eq("difficulty", difficulty)

    result = query.execute()

    if format == "json":
        # Format for training loader
        triplets = [
            {
                "anchor": {
                    "product_id": t["anchor_product_id"],
                    "frame_idx": t["anchor_frame_idx"],
                },
                "positive": {
                    "product_id": t["positive_product_id"],
                    "frame_idx": t["positive_frame_idx"],
                },
                "negative": {
                    "product_id": t["negative_product_id"],
                    "frame_idx": t["negative_frame_idx"],
                },
                "difficulty": t["difficulty"],
                "margin": t.get("margin", t["anchor_positive_sim"] - t["anchor_negative_sim"]),
            }
            for t in result.data
        ]
        return {"triplets": triplets, "count": len(triplets)}

    elif format == "csv":
        import io
        import csv
        from fastapi.responses import StreamingResponse

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "anchor_product_id", "anchor_frame_idx",
            "positive_product_id", "positive_frame_idx",
            "negative_product_id", "negative_frame_idx",
            "difficulty", "margin"
        ])
        writer.writeheader()

        for t in result.data:
            writer.writerow({
                "anchor_product_id": t["anchor_product_id"],
                "anchor_frame_idx": t["anchor_frame_idx"],
                "positive_product_id": t["positive_product_id"],
                "positive_frame_idx": t["positive_frame_idx"],
                "negative_product_id": t["negative_product_id"],
                "negative_frame_idx": t["negative_frame_idx"],
                "difficulty": t["difficulty"],
                "margin": t.get("margin", t["anchor_positive_sim"] - t["anchor_negative_sim"]),
            })

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=triplets_{run_id}.csv"}
        )


# ===========================================
# Feedback Endpoints
# ===========================================

class MatchingFeedbackRequest(BaseModel):
    """Request to submit matching feedback."""
    cutout_id: Optional[str] = None
    cutout_image_url: Optional[str] = None
    predicted_product_id: str
    predicted_similarity: float
    model_id: Optional[str] = None
    collection_name: Optional[str] = None
    feedback_type: Literal["correct", "wrong", "uncertain"]
    correct_product_id: Optional[str] = None  # Required if feedback_type is 'wrong'


@router.post("/feedback")
async def submit_matching_feedback(
    request: MatchingFeedbackRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Submit feedback for a matching result (for active learning)."""
    if request.feedback_type == "wrong" and not request.correct_product_id:
        raise HTTPException(400, "correct_product_id is required when feedback_type is 'wrong'")

    data = {
        "cutout_id": request.cutout_id,
        "cutout_image_url": request.cutout_image_url,
        "predicted_product_id": request.predicted_product_id,
        "predicted_similarity": request.predicted_similarity,
        "model_id": request.model_id,
        "collection_name": request.collection_name,
        "feedback_type": request.feedback_type,
        "correct_product_id": request.correct_product_id,
    }

    result = db.client.table("matching_feedback").insert(data).execute()

    if not result.data:
        raise HTTPException(500, "Failed to save feedback")

    return {"status": "recorded", "id": result.data[0]["id"]}


@router.get("/feedback/stats")
async def get_feedback_stats(
    model_id: Optional[str] = None,
    db: SupabaseService = Depends(get_supabase),
):
    """Get feedback statistics for active learning."""
    query = db.client.table("matching_feedback").select("feedback_type")

    if model_id:
        query = query.eq("model_id", model_id)

    result = query.execute()

    if not result.data:
        return {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "uncertain": 0,
            "accuracy": 0.0,
        }

    total = len(result.data)
    correct = sum(1 for r in result.data if r["feedback_type"] == "correct")
    wrong = sum(1 for r in result.data if r["feedback_type"] == "wrong")
    uncertain = sum(1 for r in result.data if r["feedback_type"] == "uncertain")

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "uncertain": uncertain,
        "accuracy": correct / total if total > 0 else 0.0,
    }


@router.get("/feedback/hard-examples")
async def get_hard_examples_from_feedback(
    model_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: SupabaseService = Depends(get_supabase),
):
    """Get wrong predictions as hard training examples."""
    # Use the database function
    result = db.client.rpc(
        "get_hard_examples_from_feedback",
        {"p_model_id": model_id, "p_limit": limit}
    ).execute()

    return {
        "hard_examples": result.data or [],
        "count": len(result.data) if result.data else 0,
    }
