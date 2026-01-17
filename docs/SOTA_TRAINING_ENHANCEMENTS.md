# SOTA Training Enhancements Plan

## Overview

Bu doküman, BuyBuddy AI product recognition modelinin SOTA (State-of-the-Art) seviyesine çıkarılması için gerekli geliştirmeleri kapsar. Mevcut sistemle tam uyumlu olacak şekilde tasarlanmıştır.

**Hedef:** Synthetic → Real domain adaptation ile yüksek doğruluklu ürün tanıma

---

## Current System Analysis

### Mevcut Yapı
| Component | Current State | Location |
|-----------|---------------|----------|
| Loss Function | ArcFace (CrossEntropy tabanlı) | `workers/training/src/losses.py` |
| Batch Sampling | BalancedBatchSampler, DomainBalancedSampler | `workers/training/src/dataset.py` |
| Models | DINOv2, DINOv3, CLIP | `packages/bb-models/` |
| Labels | product_id based (configurable) | `training_runs.label_config` |
| Domain Info | brand_name field | `products.brand_name` |

### Mevcut Eksiklikler
1. ❌ Triplet Loss yok
2. ❌ Hard Negative Mining yok
3. ❌ Synthetic/Real domain separation yok
4. ❌ Curriculum Learning yok
5. ❌ Online Hard Mining yok

---

## Enhancement Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOTA TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Triplet    │    │    Data      │    │   Training   │      │
│  │   Mining     │───▶│   Loader     │───▶│    Loop      │      │
│  │   Service    │    │  (P-K Batch) │    │ (Multi-Loss) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Hard Neg    │    │   Domain     │    │  Curriculum  │      │
│  │  Database    │    │  Balancing   │    │   Scheduler  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Triplet Mining System

### 1.1 Database Schema

```sql
-- Migration: 014_triplet_mining.sql

-- Triplet mining runs
CREATE TABLE triplet_mining_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,

    -- Source
    dataset_id UUID REFERENCES datasets(id),
    embedding_model_id UUID REFERENCES embedding_models(id) NOT NULL,
    collection_name TEXT NOT NULL,  -- Qdrant collection used

    -- Config
    hard_negative_threshold REAL DEFAULT 0.7,  -- Similarity threshold
    positive_threshold REAL DEFAULT 0.9,
    max_triplets_per_anchor INTEGER DEFAULT 10,
    include_cross_domain BOOLEAN DEFAULT true,  -- Synthetic-Real pairs

    -- Stats
    total_anchors INTEGER,
    total_triplets INTEGER,
    hard_triplets INTEGER,
    semi_hard_triplets INTEGER,
    cross_domain_triplets INTEGER,

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'running', 'completed', 'failed'
    )),
    error_message TEXT,

    -- Output
    output_url TEXT,  -- S3 URL to triplets file

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual triplets (for UI display and analysis)
CREATE TABLE mined_triplets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mining_run_id UUID REFERENCES triplet_mining_runs(id) ON DELETE CASCADE,

    -- Triplet components
    anchor_product_id TEXT NOT NULL,
    positive_product_id TEXT NOT NULL,
    negative_product_id TEXT NOT NULL,

    -- Frame indices
    anchor_frame_idx INTEGER DEFAULT 0,
    positive_frame_idx INTEGER DEFAULT 0,
    negative_frame_idx INTEGER DEFAULT 0,

    -- Similarities
    anchor_positive_sim REAL NOT NULL,
    anchor_negative_sim REAL NOT NULL,
    margin REAL GENERATED ALWAYS AS (anchor_positive_sim - anchor_negative_sim) STORED,

    -- Classification
    difficulty TEXT CHECK (difficulty IN ('hard', 'semi_hard', 'easy')),
    is_cross_domain BOOLEAN DEFAULT false,  -- Anchor synthetic, negative real

    -- Domain info
    anchor_domain TEXT,  -- 'synthetic' or 'real'
    positive_domain TEXT,
    negative_domain TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_mined_triplets_run ON mined_triplets(mining_run_id);
CREATE INDEX idx_mined_triplets_difficulty ON mined_triplets(difficulty);
CREATE INDEX idx_mined_triplets_cross_domain ON mined_triplets(is_cross_domain);

-- Link triplet mining to training runs
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS triplet_mining_run_id UUID REFERENCES triplet_mining_runs(id);

ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS triplet_config JSONB DEFAULT '{
    "enabled": false,
    "hard_ratio": 0.5,
    "semi_hard_ratio": 0.3,
    "random_ratio": 0.2,
    "online_mining": true
}'::jsonb;
```

### 1.2 Backend API - Triplet Mining

```python
# apps/api/src/api/v1/triplets.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import numpy as np

router = APIRouter(prefix="/triplets", tags=["triplets"])

class TripletMiningConfig(BaseModel):
    dataset_id: Optional[str] = None
    embedding_model_id: str
    collection_name: str
    hard_negative_threshold: float = 0.7
    positive_threshold: float = 0.9
    max_triplets_per_anchor: int = 10
    include_cross_domain: bool = True

class TripletMiningResponse(BaseModel):
    id: str
    status: str
    message: str

@router.post("/mine", response_model=TripletMiningResponse)
async def start_triplet_mining(
    config: TripletMiningConfig,
    background_tasks: BackgroundTasks,
    db = Depends(get_supabase),
    qdrant = Depends(get_qdrant),
):
    """Start a triplet mining job."""

    # Create mining run record
    run_data = {
        "dataset_id": config.dataset_id,
        "embedding_model_id": config.embedding_model_id,
        "collection_name": config.collection_name,
        "hard_negative_threshold": config.hard_negative_threshold,
        "positive_threshold": config.positive_threshold,
        "max_triplets_per_anchor": config.max_triplets_per_anchor,
        "include_cross_domain": config.include_cross_domain,
        "status": "pending",
    }

    result = await db.create_triplet_mining_run(run_data)
    run_id = result["id"]

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
        message="Triplet mining started"
    )

async def mine_triplets_task(
    run_id: str,
    config: TripletMiningConfig,
    db,
    qdrant,
):
    """Background task for triplet mining."""
    try:
        await db.update_triplet_mining_run(run_id, {"status": "running"})

        # Get all embeddings from collection
        embeddings = await qdrant.get_all_embeddings(config.collection_name)

        # Group by product_id
        product_embeddings = defaultdict(list)
        for emb in embeddings:
            product_id = emb["payload"]["product_id"]
            domain = emb["payload"].get("domain", "unknown")
            product_embeddings[product_id].append({
                "vector": emb["vector"],
                "frame_idx": emb["payload"].get("frame_index", 0),
                "domain": domain,
            })

        # Mine triplets
        triplets = []
        product_ids = list(product_embeddings.keys())

        for anchor_id in product_ids:
            anchor_data = product_embeddings[anchor_id]

            for anchor_emb in anchor_data:
                anchor_vec = np.array(anchor_emb["vector"])

                # Find positives (same product, different frame)
                positives = [
                    e for e in anchor_data
                    if e["frame_idx"] != anchor_emb["frame_idx"]
                ]

                if not positives:
                    continue

                # Find hard negatives
                hard_negatives = []
                for neg_id in product_ids:
                    if neg_id == anchor_id:
                        continue

                    for neg_emb in product_embeddings[neg_id]:
                        neg_vec = np.array(neg_emb["vector"])
                        sim = np.dot(anchor_vec, neg_vec)

                        if sim >= config.hard_negative_threshold:
                            hard_negatives.append({
                                "product_id": neg_id,
                                "frame_idx": neg_emb["frame_idx"],
                                "domain": neg_emb["domain"],
                                "similarity": float(sim),
                            })

                # Sort by similarity (hardest first)
                hard_negatives.sort(key=lambda x: x["similarity"], reverse=True)
                hard_negatives = hard_negatives[:config.max_triplets_per_anchor]

                # Create triplets
                for pos_emb in positives[:3]:  # Max 3 positives per anchor
                    pos_vec = np.array(pos_emb["vector"])
                    pos_sim = float(np.dot(anchor_vec, pos_vec))

                    for neg in hard_negatives:
                        # Classify difficulty
                        margin = pos_sim - neg["similarity"]
                        if margin < 0.1:
                            difficulty = "hard"
                        elif margin < 0.3:
                            difficulty = "semi_hard"
                        else:
                            difficulty = "easy"

                        # Check cross-domain
                        is_cross_domain = (
                            anchor_emb["domain"] == "synthetic" and
                            neg["domain"] == "real"
                        )

                        triplets.append({
                            "mining_run_id": run_id,
                            "anchor_product_id": anchor_id,
                            "positive_product_id": anchor_id,
                            "negative_product_id": neg["product_id"],
                            "anchor_frame_idx": anchor_emb["frame_idx"],
                            "positive_frame_idx": pos_emb["frame_idx"],
                            "negative_frame_idx": neg["frame_idx"],
                            "anchor_positive_sim": pos_sim,
                            "anchor_negative_sim": neg["similarity"],
                            "difficulty": difficulty,
                            "is_cross_domain": is_cross_domain,
                            "anchor_domain": anchor_emb["domain"],
                            "positive_domain": pos_emb["domain"],
                            "negative_domain": neg["domain"],
                        })

        # Save triplets to database
        await db.batch_insert_triplets(triplets)

        # Update run stats
        stats = {
            "status": "completed",
            "total_anchors": len(product_ids),
            "total_triplets": len(triplets),
            "hard_triplets": len([t for t in triplets if t["difficulty"] == "hard"]),
            "semi_hard_triplets": len([t for t in triplets if t["difficulty"] == "semi_hard"]),
            "cross_domain_triplets": len([t for t in triplets if t["is_cross_domain"]]),
            "completed_at": "NOW()",
        }
        await db.update_triplet_mining_run(run_id, stats)

    except Exception as e:
        await db.update_triplet_mining_run(run_id, {
            "status": "failed",
            "error_message": str(e),
        })

@router.get("/runs")
async def list_mining_runs(db = Depends(get_supabase)):
    """List all triplet mining runs."""
    return await db.list_triplet_mining_runs()

@router.get("/runs/{run_id}")
async def get_mining_run(run_id: str, db = Depends(get_supabase)):
    """Get details of a mining run."""
    return await db.get_triplet_mining_run(run_id)

@router.get("/runs/{run_id}/triplets")
async def get_triplets(
    run_id: str,
    difficulty: Optional[str] = None,
    cross_domain_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    db = Depends(get_supabase),
):
    """Get triplets from a mining run."""
    return await db.get_triplets(
        run_id=run_id,
        difficulty=difficulty,
        cross_domain_only=cross_domain_only,
        limit=limit,
        offset=offset,
    )

@router.get("/runs/{run_id}/stats")
async def get_triplet_stats(run_id: str, db = Depends(get_supabase)):
    """Get statistics for a mining run."""
    triplets = await db.get_all_triplets(run_id)

    return {
        "total": len(triplets),
        "by_difficulty": {
            "hard": len([t for t in triplets if t["difficulty"] == "hard"]),
            "semi_hard": len([t for t in triplets if t["difficulty"] == "semi_hard"]),
            "easy": len([t for t in triplets if t["difficulty"] == "easy"]),
        },
        "cross_domain": len([t for t in triplets if t["is_cross_domain"]]),
        "margin_distribution": {
            "mean": np.mean([t["margin"] for t in triplets]),
            "std": np.std([t["margin"] for t in triplets]),
            "min": min([t["margin"] for t in triplets]),
            "max": max([t["margin"] for t in triplets]),
        },
    }
```

### 1.3 Frontend - Triplets Page

```tsx
// apps/web/src/app/triplets/page.tsx

"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Triangle, Play, Download, Loader2, Info, Database, Eye, BarChart3 } from "lucide-react";
import { TripletViewer } from "./components/TripletViewer";
import { MiningStats } from "./components/MiningStats";

export default function TripletsPage() {
  const queryClient = useQueryClient();
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedCollection, setSelectedCollection] = useState<string>("");
  const [hardNegativeThreshold, setHardNegativeThreshold] = useState([0.7]);
  const [positiveThreshold, setPositiveThreshold] = useState([0.9]);
  const [maxTripletsPerAnchor, setMaxTripletsPerAnchor] = useState([10]);
  const [includeCrossDomain, setIncludeCrossDomain] = useState(true);

  // Fetch datasets
  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Fetch models
  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ["embedding-models"],
    queryFn: () => apiClient.getEmbeddingModels(),
  });

  // Fetch collections
  const { data: collections, isLoading: collectionsLoading } = useQuery({
    queryKey: ["qdrant-collections"],
    queryFn: () => apiClient.getQdrantCollections(),
  });

  // Fetch mining runs
  const { data: miningRuns, isLoading: runsLoading } = useQuery({
    queryKey: ["triplet-mining-runs"],
    queryFn: () => apiClient.getTripletMiningRuns(),
  });

  // Start mining mutation
  const startMining = useMutation({
    mutationFn: (config: TripletMiningConfig) => apiClient.startTripletMining(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["triplet-mining-runs"] });
    },
  });

  const handleStartMining = () => {
    startMining.mutate({
      dataset_id: selectedDataset || undefined,
      embedding_model_id: selectedModel,
      collection_name: selectedCollection,
      hard_negative_threshold: hardNegativeThreshold[0],
      positive_threshold: positiveThreshold[0],
      max_triplets_per_anchor: maxTripletsPerAnchor[0],
      include_cross_domain: includeCrossDomain,
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Hard Triplet Mining</h1>
          <p className="text-muted-foreground">
            Find confusing product pairs to improve model training
          </p>
        </div>
      </div>

      <Tabs defaultValue="mine" className="space-y-4">
        <TabsList>
          <TabsTrigger value="mine">
            <Triangle className="h-4 w-4 mr-2" />
            Mine Triplets
          </TabsTrigger>
          <TabsTrigger value="runs">
            <Database className="h-4 w-4 mr-2" />
            Mining Runs
          </TabsTrigger>
          <TabsTrigger value="viewer">
            <Eye className="h-4 w-4 mr-2" />
            Triplet Viewer
          </TabsTrigger>
          <TabsTrigger value="stats">
            <BarChart3 className="h-4 w-4 mr-2" />
            Statistics
          </TabsTrigger>
        </TabsList>

        {/* Mine Tab */}
        <TabsContent value="mine" className="space-y-6">
          {/* Info Card */}
          <Card className="bg-blue-50 border-blue-200">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-blue-800">
                <Info className="h-4 w-4" />
                What is Hard Triplet Mining?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-blue-700 text-sm">
              <p>
                Hard triplet mining finds products that look similar but are different (hard negatives).
                Training on these challenging examples significantly improves model accuracy.
              </p>
              <ul className="mt-2 space-y-1 list-disc list-inside">
                <li><strong>Anchor:</strong> Reference product embedding</li>
                <li><strong>Positive:</strong> Same product, different view/frame</li>
                <li><strong>Hard Negative:</strong> Different product that looks similar</li>
              </ul>
              <p className="mt-2">
                <strong>Cross-Domain Mining:</strong> Finds synthetic products confused with real products,
                critical for synthetic→real domain adaptation.
              </p>
            </CardContent>
          </Card>

          {/* Configuration */}
          <div className="grid grid-cols-2 gap-6">
            {/* Source Selection */}
            <Card>
              <CardHeader>
                <CardTitle>Data Source</CardTitle>
                <CardDescription>
                  Select embeddings to mine triplets from
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Embedding Model</Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select model..." />
                    </SelectTrigger>
                    <SelectContent>
                      {models?.map((model) => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Qdrant Collection</Label>
                  <Select value={selectedCollection} onValueChange={setSelectedCollection}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select collection..." />
                    </SelectTrigger>
                    <SelectContent>
                      {collections?.map((col) => (
                        <SelectItem key={col.name} value={col.name}>
                          <div className="flex items-center gap-2">
                            <Database className="h-4 w-4" />
                            {col.name}
                            <Badge variant="secondary">{col.vectors_count} vectors</Badge>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Dataset (Optional)</Label>
                  <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                    <SelectTrigger>
                      <SelectValue placeholder="All products" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All products</SelectItem>
                      {datasets?.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Threshold Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Mining Parameters</CardTitle>
                <CardDescription>
                  Configure similarity thresholds for triplet selection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <Label>Hard Negative Threshold</Label>
                    <span className="text-sm font-medium">{hardNegativeThreshold[0]}</span>
                  </div>
                  <Slider
                    value={hardNegativeThreshold}
                    onValueChange={setHardNegativeThreshold}
                    min={0.5}
                    max={0.95}
                    step={0.05}
                  />
                  <p className="text-xs text-muted-foreground">
                    Products with similarity above this are "hard" negatives
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <Label>Positive Threshold</Label>
                    <span className="text-sm font-medium">{positiveThreshold[0]}</span>
                  </div>
                  <Slider
                    value={positiveThreshold}
                    onValueChange={setPositiveThreshold}
                    min={0.8}
                    max={1.0}
                    step={0.02}
                  />
                  <p className="text-xs text-muted-foreground">
                    Same-product pairs must have similarity above this
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <Label>Max Triplets per Anchor</Label>
                    <span className="text-sm font-medium">{maxTripletsPerAnchor[0]}</span>
                  </div>
                  <Slider
                    value={maxTripletsPerAnchor}
                    onValueChange={setMaxTripletsPerAnchor}
                    min={1}
                    max={50}
                    step={1}
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="cross-domain"
                    checked={includeCrossDomain}
                    onChange={(e) => setIncludeCrossDomain(e.target.checked)}
                    className="rounded"
                  />
                  <Label htmlFor="cross-domain">
                    Include Cross-Domain Triplets (Synthetic → Real)
                  </Label>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Action Button */}
          <Card>
            <CardContent className="pt-6">
              <Button
                onClick={handleStartMining}
                disabled={!selectedModel || !selectedCollection || startMining.isPending}
                className="w-full"
                size="lg"
              >
                {startMining.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Mining in Progress...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Triplet Mining
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Mining Runs Tab */}
        <TabsContent value="runs">
          <MiningRunsList runs={miningRuns} isLoading={runsLoading} />
        </TabsContent>

        {/* Triplet Viewer Tab */}
        <TabsContent value="viewer">
          <TripletViewer miningRuns={miningRuns} />
        </TabsContent>

        {/* Statistics Tab */}
        <TabsContent value="stats">
          <MiningStats miningRuns={miningRuns} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

---

## Phase 2: Multi-Loss Training System

### 2.1 Enhanced Loss Functions

```python
# workers/training/src/losses.py (Enhanced)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CombinedProductLoss(nn.Module):
    """
    Combined loss for SOTA product recognition.

    Components:
    1. ArcFace Loss - Classification with angular margin
    2. Triplet Loss - Metric learning with hard negatives
    3. Domain Adversarial Loss - Domain adaptation (optional)

    Args:
        num_classes: Number of product classes
        embedding_dim: Embedding dimension
        arcface_weight: Weight for ArcFace loss (default: 1.0)
        triplet_weight: Weight for Triplet loss (default: 0.5)
        domain_weight: Weight for Domain loss (default: 0.1)
        arcface_margin: Angular margin for ArcFace (default: 0.5)
        arcface_scale: Scale for ArcFace (default: 64.0)
        triplet_margin: Margin for triplet loss (default: 0.3)
        use_domain_adaptation: Enable domain adversarial training
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        arcface_weight: float = 1.0,
        triplet_weight: float = 0.5,
        domain_weight: float = 0.1,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        triplet_margin: float = 0.3,
        use_domain_adaptation: bool = True,
    ):
        super().__init__()

        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight
        self.domain_weight = domain_weight
        self.use_domain_adaptation = use_domain_adaptation

        # ArcFace Loss
        self.arcface = ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=arcface_margin,
            scale=arcface_scale,
        )

        # Triplet Loss with Online Hard Mining
        self.triplet = OnlineHardTripletLoss(margin=triplet_margin)

        # Domain Classifier (for adversarial training)
        if use_domain_adaptation:
            self.domain_classifier = nn.Sequential(
                GradientReversal(alpha=1.0),
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2),  # synthetic vs real
            )
            self.domain_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        domains: Optional[torch.Tensor] = None,
        triplet_indices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            embeddings: [B, D] normalized embeddings
            labels: [B] class labels (product_ids)
            domains: [B] domain labels (0=synthetic, 1=real)
            triplet_indices: Optional pre-mined (anchor_idx, pos_idx, neg_idx)

        Returns:
            Dictionary with total_loss and component losses
        """
        losses = {}

        # 1. ArcFace Loss
        arcface_loss = self.arcface(embeddings, labels)
        losses["arcface"] = arcface_loss

        # 2. Triplet Loss
        if triplet_indices is not None:
            # Use pre-mined triplets
            anchor_idx, pos_idx, neg_idx = triplet_indices
            triplet_loss = self.triplet.forward_with_indices(
                embeddings, anchor_idx, pos_idx, neg_idx
            )
        else:
            # Online hard mining
            triplet_loss = self.triplet(embeddings, labels)
        losses["triplet"] = triplet_loss

        # 3. Domain Adversarial Loss
        if self.use_domain_adaptation and domains is not None:
            domain_logits = self.domain_classifier(embeddings)
            domain_loss = self.domain_loss(domain_logits, domains)
            losses["domain"] = domain_loss
        else:
            domain_loss = torch.tensor(0.0, device=embeddings.device)
            losses["domain"] = domain_loss

        # Combined loss
        total_loss = (
            self.arcface_weight * arcface_loss +
            self.triplet_weight * triplet_loss +
            self.domain_weight * domain_loss
        )
        losses["total"] = total_loss

        return losses


class OnlineHardTripletLoss(nn.Module):
    """
    Triplet loss with online hard negative mining.

    Mining strategies:
    - Hard: Hardest negative (closest to anchor)
    - Semi-hard: Negatives within margin
    - Random: Random negatives for diversity
    """

    def __init__(
        self,
        margin: float = 0.3,
        hard_ratio: float = 0.5,
        semi_hard_ratio: float = 0.3,
        random_ratio: float = 0.2,
    ):
        super().__init__()
        self.margin = margin
        self.hard_ratio = hard_ratio
        self.semi_hard_ratio = semi_hard_ratio
        self.random_ratio = random_ratio

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with online mining.

        Args:
            embeddings: [B, D] normalized embeddings
            labels: [B] class labels

        Returns:
            Triplet loss value
        """
        # Compute pairwise distances
        dist_mat = self._pairwise_distances(embeddings)

        # Get valid triplet mask
        batch_size = embeddings.size(0)
        labels = labels.unsqueeze(0)

        # Positive mask: same label, different index
        pos_mask = (labels == labels.T) & ~torch.eye(batch_size, device=embeddings.device).bool()

        # Negative mask: different label
        neg_mask = labels != labels.T

        # Mine triplets
        triplet_loss = torch.tensor(0.0, device=embeddings.device)
        num_triplets = 0

        for anchor_idx in range(batch_size):
            # Get positive indices
            pos_indices = torch.where(pos_mask[anchor_idx])[0]
            if len(pos_indices) == 0:
                continue

            # Get negative indices
            neg_indices = torch.where(neg_mask[anchor_idx])[0]
            if len(neg_indices) == 0:
                continue

            # Anchor-positive distances
            ap_dists = dist_mat[anchor_idx, pos_indices]

            # Anchor-negative distances
            an_dists = dist_mat[anchor_idx, neg_indices]

            # Mine based on ratios
            num_hard = int(len(neg_indices) * self.hard_ratio)
            num_semi = int(len(neg_indices) * self.semi_hard_ratio)

            # Hard negatives: smallest distance
            hard_neg_indices = neg_indices[torch.argsort(an_dists)[:num_hard]]

            # Semi-hard negatives: within margin
            max_ap = ap_dists.max()
            semi_hard_mask = (an_dists > max_ap) & (an_dists < max_ap + self.margin)
            semi_hard_neg_indices = neg_indices[semi_hard_mask][:num_semi]

            # Random negatives
            remaining_indices = list(set(neg_indices.tolist()) -
                                    set(hard_neg_indices.tolist()) -
                                    set(semi_hard_neg_indices.tolist()))
            num_random = min(len(remaining_indices), len(neg_indices) - num_hard - num_semi)
            random_neg_indices = torch.tensor(
                remaining_indices[:num_random],
                device=embeddings.device
            )

            # Combine negatives
            all_neg_indices = torch.cat([
                hard_neg_indices,
                semi_hard_neg_indices,
                random_neg_indices,
            ])

            # Compute triplet loss for this anchor
            for pos_idx in pos_indices:
                ap_dist = dist_mat[anchor_idx, pos_idx]
                for neg_idx in all_neg_indices:
                    an_dist = dist_mat[anchor_idx, neg_idx]
                    loss = F.relu(ap_dist - an_dist + self.margin)
                    triplet_loss += loss
                    num_triplets += 1

        if num_triplets > 0:
            triplet_loss = triplet_loss / num_triplets

        return triplet_loss

    def forward_with_indices(
        self,
        embeddings: torch.Tensor,
        anchor_indices: torch.Tensor,
        pos_indices: torch.Tensor,
        neg_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with pre-mined indices.
        """
        anchor_emb = embeddings[anchor_indices]
        pos_emb = embeddings[pos_indices]
        neg_emb = embeddings[neg_indices]

        ap_dist = F.pairwise_distance(anchor_emb, pos_emb)
        an_dist = F.pairwise_distance(anchor_emb, neg_emb)

        loss = F.relu(ap_dist - an_dist + self.margin)
        return loss.mean()

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)
        return torch.sqrt(distances + 1e-16)


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for Domain Adversarial Training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalModule(nn.Module):
    """Wrapper module for Gradient Reversal."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)
```

### 2.2 Enhanced Training Configuration

```python
# workers/training/src/config.py (Enhanced)

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class LossType(str, Enum):
    ARCFACE = "arcface"
    TRIPLET = "triplet"
    COMBINED = "combined"
    DOMAIN_ADAPTIVE = "domain_adaptive"


class MiningStrategy(str, Enum):
    ONLINE = "online"  # Mine during training
    PRECOMPUTED = "precomputed"  # Use pre-mined triplets
    HYBRID = "hybrid"  # Combine both


class SamplingStrategy(str, Enum):
    RANDOM = "random"
    BALANCED = "balanced"
    PK_SAMPLER = "pk_sampler"
    DOMAIN_BALANCED = "domain_balanced"


@dataclass
class LossConfig:
    """Configuration for loss functions."""

    loss_type: LossType = LossType.COMBINED

    # ArcFace settings
    arcface_weight: float = 1.0
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0

    # Triplet settings
    triplet_weight: float = 0.5
    triplet_margin: float = 0.3

    # Mining settings
    mining_strategy: MiningStrategy = MiningStrategy.ONLINE
    hard_ratio: float = 0.5
    semi_hard_ratio: float = 0.3
    random_ratio: float = 0.2

    # Domain adaptation
    use_domain_adaptation: bool = True
    domain_weight: float = 0.1


@dataclass
class SamplingConfig:
    """Configuration for batch sampling."""

    strategy: SamplingStrategy = SamplingStrategy.PK_SAMPLER

    # P-K Sampler settings
    products_per_batch: int = 8  # P
    samples_per_product: int = 4  # K

    # Domain balancing
    domains_per_batch: int = 4
    balance_synthetic_real: bool = True
    synthetic_ratio: float = 0.5  # Ratio of synthetic samples


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    enabled: bool = True

    # Phases (epochs)
    warmup_epochs: int = 2
    easy_epochs: int = 5
    hard_epochs: int = 10
    finetune_epochs: int = 3

    # Difficulty thresholds
    easy_margin: float = 0.5
    hard_margin: float = 0.2

    # Learning rate schedule
    warmup_lr_mult: float = 0.1
    finetune_lr_mult: float = 0.01


@dataclass
class DomainAdaptationConfig:
    """Configuration for domain adaptation."""

    enabled: bool = True

    # MixUp augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.4

    # Domain adversarial
    use_adversarial: bool = True
    adversarial_weight: float = 0.1
    gradient_reversal_alpha: float = 1.0

    # Cross-domain triplets
    cross_domain_triplet_ratio: float = 0.3


@dataclass
class SOTATrainingConfig:
    """Complete SOTA training configuration."""

    # Base model
    base_model_type: str = "dinov3-base"

    # Loss
    loss: LossConfig = field(default_factory=LossConfig)

    # Sampling
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Curriculum
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Domain adaptation
    domain_adaptation: DomainAdaptationConfig = field(default_factory=DomainAdaptationConfig)

    # Pre-mined triplets
    triplet_mining_run_id: Optional[str] = None

    # Standard training params
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 20

    # LLRD (Layer-wise Learning Rate Decay)
    use_llrd: bool = True
    llrd_decay: float = 0.9

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    save_every_n_epochs: int = 1
    keep_best_n: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOTATrainingConfig":
        """Create from dictionary."""
        return cls(
            base_model_type=data.get("base_model_type", "dinov3-base"),
            loss=LossConfig(**data.get("loss", {})),
            sampling=SamplingConfig(**data.get("sampling", {})),
            curriculum=CurriculumConfig(**data.get("curriculum", {})),
            domain_adaptation=DomainAdaptationConfig(**data.get("domain_adaptation", {})),
            triplet_mining_run_id=data.get("triplet_mining_run_id"),
            learning_rate=data.get("learning_rate", 1e-4),
            weight_decay=data.get("weight_decay", 0.01),
            batch_size=data.get("batch_size", 32),
            num_epochs=data.get("num_epochs", 20),
            use_llrd=data.get("use_llrd", True),
            llrd_decay=data.get("llrd_decay", 0.9),
            use_amp=data.get("use_amp", True),
            save_every_n_epochs=data.get("save_every_n_epochs", 1),
            keep_best_n=data.get("keep_best_n", 3),
        )
```

---

## Phase 3: Smart Batch Sampling

### 3.1 P-K Sampler with Domain Balancing

```python
# workers/training/src/samplers.py

import random
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple
import torch
from torch.utils.data import Sampler


class PKDomainSampler(Sampler):
    """
    P-K Sampler with Domain Balancing.

    Each batch contains:
    - P different products
    - K samples per product
    - Balanced synthetic/real ratio

    This ensures each batch has:
    - Multiple views of same product (for positives)
    - Multiple products (for negatives)
    - Mix of synthetic and real (for domain adaptation)
    """

    def __init__(
        self,
        labels: List[int],
        domains: List[int],  # 0=synthetic, 1=real
        products_per_batch: int = 8,
        samples_per_product: int = 4,
        synthetic_ratio: float = 0.5,
        drop_last: bool = True,
    ):
        """
        Args:
            labels: List of product_id indices for each sample
            domains: List of domain labels (0=synthetic, 1=real)
            products_per_batch: Number of different products per batch (P)
            samples_per_product: Number of samples per product (K)
            synthetic_ratio: Target ratio of synthetic samples
            drop_last: Drop last incomplete batch
        """
        self.labels = labels
        self.domains = domains
        self.products_per_batch = products_per_batch
        self.samples_per_product = samples_per_product
        self.synthetic_ratio = synthetic_ratio
        self.drop_last = drop_last

        self.batch_size = products_per_batch * samples_per_product

        # Build index mappings
        self._build_indices()

    def _build_indices(self):
        """Build product -> sample indices mapping."""
        # Product to indices
        self.product_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.product_to_indices[label].append(idx)

        # Separate by domain
        self.synthetic_products = set()
        self.real_products = set()

        for idx, (label, domain) in enumerate(zip(self.labels, self.domains)):
            if domain == 0:
                self.synthetic_products.add(label)
            else:
                self.real_products.add(label)

        # Filter products with enough samples
        self.valid_synthetic = [
            p for p in self.synthetic_products
            if len(self.product_to_indices[p]) >= self.samples_per_product
        ]
        self.valid_real = [
            p for p in self.real_products
            if len(self.product_to_indices[p]) >= self.samples_per_product
        ]

        self.all_products = list(set(self.valid_synthetic + self.valid_real))

        print(f"PKDomainSampler initialized:")
        print(f"  Synthetic products: {len(self.valid_synthetic)}")
        print(f"  Real products: {len(self.valid_real)}")
        print(f"  Batch size: {self.batch_size} (P={self.products_per_batch}, K={self.samples_per_product})")

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches."""
        # Shuffle products
        synthetic_products = self.valid_synthetic.copy()
        real_products = self.valid_real.copy()
        random.shuffle(synthetic_products)
        random.shuffle(real_products)

        # Calculate products per domain per batch
        num_synthetic = int(self.products_per_batch * self.synthetic_ratio)
        num_real = self.products_per_batch - num_synthetic

        # Adjust if not enough products
        if len(synthetic_products) < num_synthetic:
            num_synthetic = len(synthetic_products)
            num_real = self.products_per_batch - num_synthetic
        if len(real_products) < num_real:
            num_real = len(real_products)
            num_synthetic = self.products_per_batch - num_real

        syn_ptr = 0
        real_ptr = 0

        while True:
            batch = []
            batch_products = []

            # Sample synthetic products
            for _ in range(num_synthetic):
                if syn_ptr >= len(synthetic_products):
                    random.shuffle(synthetic_products)
                    syn_ptr = 0
                batch_products.append(synthetic_products[syn_ptr])
                syn_ptr += 1

            # Sample real products
            for _ in range(num_real):
                if real_ptr >= len(real_products):
                    random.shuffle(real_products)
                    real_ptr = 0
                batch_products.append(real_products[real_ptr])
                real_ptr += 1

            # Sample K instances per product
            for product in batch_products:
                indices = self.product_to_indices[product]
                if len(indices) >= self.samples_per_product:
                    sampled = random.sample(indices, self.samples_per_product)
                else:
                    # With replacement if not enough
                    sampled = random.choices(indices, k=self.samples_per_product)
                batch.extend(sampled)

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch

            # Check if exhausted
            if syn_ptr >= len(synthetic_products) and real_ptr >= len(real_products):
                break

    def __len__(self) -> int:
        """Approximate number of batches."""
        total_products = len(self.all_products)
        return total_products // self.products_per_batch


class CurriculumSampler(Sampler):
    """
    Curriculum Learning Sampler.

    Progressively increases difficulty:
    1. Warmup: Random easy samples
    2. Easy: Wide margin triplets
    3. Hard: Tight margin triplets
    4. Finetune: Focus on hardest cases
    """

    def __init__(
        self,
        dataset,
        triplet_difficulties: List[str],  # 'easy', 'semi_hard', 'hard'
        current_phase: str = "warmup",
        batch_size: int = 32,
    ):
        self.dataset = dataset
        self.difficulties = triplet_difficulties
        self.current_phase = current_phase
        self.batch_size = batch_size

        # Index by difficulty
        self.easy_indices = [
            i for i, d in enumerate(triplet_difficulties) if d == "easy"
        ]
        self.semi_hard_indices = [
            i for i, d in enumerate(triplet_difficulties) if d == "semi_hard"
        ]
        self.hard_indices = [
            i for i, d in enumerate(triplet_difficulties) if d == "hard"
        ]

    def set_phase(self, phase: str):
        """Set curriculum phase."""
        self.current_phase = phase

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches based on current phase."""
        if self.current_phase == "warmup":
            # Random sampling
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
        elif self.current_phase == "easy":
            # Easy + some semi-hard
            indices = self.easy_indices + self.semi_hard_indices[:len(self.easy_indices) // 2]
            random.shuffle(indices)
        elif self.current_phase == "hard":
            # All difficulties with emphasis on hard
            indices = (
                self.hard_indices * 2 +  # Double weight
                self.semi_hard_indices +
                self.easy_indices[:len(self.hard_indices)]
            )
            random.shuffle(indices)
        elif self.current_phase == "finetune":
            # Focus on hardest
            indices = self.hard_indices + self.semi_hard_indices[:len(self.hard_indices) // 2]
            random.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
```

---

## Phase 4: Training Loop Integration

### 4.1 Enhanced Trainer

```python
# workers/training/src/trainer.py (Enhanced)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any
import wandb
from tqdm import tqdm

from .config import SOTATrainingConfig, CurriculumConfig
from .losses import CombinedProductLoss
from .samplers import PKDomainSampler, CurriculumSampler
from .dataset import ProductDataset


class SOTATrainer:
    """
    SOTA Trainer for Product Recognition.

    Features:
    - Multi-loss training (ArcFace + Triplet + Domain)
    - P-K batch sampling with domain balancing
    - Curriculum learning
    - Layer-wise learning rate decay
    - Mixed precision training
    """

    def __init__(
        self,
        model: nn.Module,
        config: SOTATrainingConfig,
        train_dataset: ProductDataset,
        val_dataset: ProductDataset,
        device: torch.device,
        checkpoint_dir: str,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Setup loss
        self.loss_fn = CombinedProductLoss(
            num_classes=train_dataset.num_classes,
            embedding_dim=model.embedding_dim,
            arcface_weight=config.loss.arcface_weight,
            triplet_weight=config.loss.triplet_weight,
            domain_weight=config.loss.domain_weight,
            arcface_margin=config.loss.arcface_margin,
            arcface_scale=config.loss.arcface_scale,
            triplet_margin=config.loss.triplet_margin,
            use_domain_adaptation=config.domain_adaptation.enabled,
        ).to(device)

        # Setup optimizer with LLRD
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup data loaders
        self.train_loader = self._setup_train_loader()
        self.val_loader = self._setup_val_loader()

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Curriculum state
        self.curriculum_phase = "warmup"
        self.current_epoch = 0

        # Best metrics
        self.best_val_loss = float('inf')
        self.best_recall_at_1 = 0.0

    def _setup_optimizer(self):
        """Setup optimizer with LLRD."""
        if self.config.use_llrd:
            # Get parameter groups with layer-wise learning rates
            param_groups = self._get_llrd_param_groups()
        else:
            param_groups = [{'params': self.model.parameters()}]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _get_llrd_param_groups(self):
        """Get parameter groups with layer-wise learning rate decay."""
        # Get all named parameters
        param_groups = []

        # Backbone layers (decay from top to bottom)
        backbone_layers = []
        for name, param in self.model.backbone.named_parameters():
            layer_num = self._get_layer_num(name)
            backbone_layers.append((name, param, layer_num))

        # Sort by layer number (deepest first)
        max_layer = max(l[2] for l in backbone_layers) if backbone_layers else 0

        for name, param, layer_num in backbone_layers:
            # Decay factor: deeper layers have higher LR
            decay_factor = self.config.llrd_decay ** (max_layer - layer_num)
            param_groups.append({
                'params': [param],
                'lr': self.config.learning_rate * decay_factor,
                'name': name,
            })

        # Head layers (full learning rate)
        for name, param in self.model.head.named_parameters():
            param_groups.append({
                'params': [param],
                'lr': self.config.learning_rate,
                'name': f'head.{name}',
            })

        # Loss function parameters
        for name, param in self.loss_fn.named_parameters():
            param_groups.append({
                'params': [param],
                'lr': self.config.learning_rate,
                'name': f'loss.{name}',
            })

        return param_groups

    def _get_layer_num(self, name: str) -> int:
        """Extract layer number from parameter name."""
        import re
        match = re.search(r'layer(\d+)|block(\d+)|encoder\.(\d+)', name)
        if match:
            return int(match.group(1) or match.group(2) or match.group(3))
        return 0

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.curriculum.enabled:
            # Curriculum-aware scheduler
            total_steps = len(self.train_loader) * self.config.num_epochs
            warmup_steps = len(self.train_loader) * self.config.curriculum.warmup_epochs

            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return self.config.curriculum.warmup_lr_mult + \
                           (1.0 - self.config.curriculum.warmup_lr_mult) * step / warmup_steps
                else:
                    # Cosine decay
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_loader) * self.config.num_epochs,
            )

    def _setup_train_loader(self):
        """Setup training data loader with P-K sampling."""
        labels = [self.train_dataset.product_id_to_idx[p["id"]]
                  for p in self.train_dataset.products]
        domains = [1 if p.get("is_real", True) else 0
                   for p in self.train_dataset.products]

        sampler = PKDomainSampler(
            labels=labels,
            domains=domains,
            products_per_batch=self.config.sampling.products_per_batch,
            samples_per_product=self.config.sampling.samples_per_product,
            synthetic_ratio=self.config.sampling.synthetic_ratio,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

    def _setup_val_loader(self):
        """Setup validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def train(self):
        """Run full training loop."""
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Update curriculum phase
            self._update_curriculum_phase(epoch)

            # Train epoch
            train_metrics = self._train_epoch()

            # Validate
            val_metrics = self._validate()

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save checkpoint
            is_best = val_metrics['recall@1'] > self.best_recall_at_1
            if is_best:
                self.best_recall_at_1 = val_metrics['recall@1']
                self.best_val_loss = val_metrics['loss']

            self._save_checkpoint(epoch, is_best)

            # Update learning rate
            self.scheduler.step()

    def _update_curriculum_phase(self, epoch: int):
        """Update curriculum learning phase."""
        if not self.config.curriculum.enabled:
            return

        cfg = self.config.curriculum

        if epoch < cfg.warmup_epochs:
            self.curriculum_phase = "warmup"
        elif epoch < cfg.warmup_epochs + cfg.easy_epochs:
            self.curriculum_phase = "easy"
        elif epoch < cfg.warmup_epochs + cfg.easy_epochs + cfg.hard_epochs:
            self.curriculum_phase = "hard"
        else:
            self.curriculum_phase = "finetune"

        print(f"Epoch {epoch}: Curriculum phase = {self.curriculum_phase}")

    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        loss_components = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move to device
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            domains = batch.get("domain", torch.ones_like(labels)).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    embeddings = self.model(images)
                    losses = self.loss_fn(embeddings, labels, domains)

                self.scaler.scale(losses["total"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings = self.model(images)
                losses = self.loss_fn(embeddings, labels, domains)
                losses["total"].backward()
                self.optimizer.step()

            # Accumulate metrics
            total_loss += losses["total"].item()
            for key, value in losses.items():
                loss_components[key] += value.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'arcface': f"{losses['arcface'].item():.4f}",
                'triplet': f"{losses['triplet'].item():.4f}",
            })

        return {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()},
        }

    def _validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        all_embeddings = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                embeddings = self.model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

        # Concatenate
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute recall@k
        recall_at_1 = self._compute_recall_at_k(all_embeddings, all_labels, k=1)
        recall_at_5 = self._compute_recall_at_k(all_embeddings, all_labels, k=5)
        recall_at_10 = self._compute_recall_at_k(all_embeddings, all_labels, k=10)

        return {
            'loss': total_loss / len(self.val_loader),
            'recall@1': recall_at_1,
            'recall@5': recall_at_5,
            'recall@10': recall_at_10,
        }

    def _compute_recall_at_k(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        k: int,
    ) -> float:
        """Compute recall@k."""
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Mask self-similarity
        sim_matrix.fill_diagonal_(-float('inf'))

        # Get top-k indices
        _, top_k_indices = sim_matrix.topk(k, dim=1)

        # Check if correct label in top-k
        correct = 0
        for i in range(len(labels)):
            query_label = labels[i]
            retrieved_labels = labels[top_k_indices[i]]
            if query_label in retrieved_labels:
                correct += 1

        return correct / len(labels)

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log metrics to wandb and console."""
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Recall@1: {val_metrics['recall@1']:.4f}")
        print(f"  Val Recall@5: {val_metrics['recall@5']:.4f}")

        if wandb.run:
            wandb.log({
                'epoch': epoch,
                'curriculum_phase': self.curriculum_phase,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
            })

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'best_recall_at_1': self.best_recall_at_1,
            'best_val_loss': self.best_val_loss,
        }

        # Save latest
        torch.save(checkpoint, f"{self.checkpoint_dir}/latest.pt")

        # Save periodic
        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            torch.save(checkpoint, f"{self.checkpoint_dir}/epoch_{epoch}.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, f"{self.checkpoint_dir}/best.pt")
```

---

## Phase 5: Database & API Integration

### 5.1 Training Runs Schema Update

```sql
-- Add to training_runs table (extends existing)

ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS sota_config JSONB DEFAULT '{
    "loss": {
        "loss_type": "combined",
        "arcface_weight": 1.0,
        "triplet_weight": 0.5,
        "domain_weight": 0.1
    },
    "sampling": {
        "strategy": "pk_sampler",
        "products_per_batch": 8,
        "samples_per_product": 4
    },
    "curriculum": {
        "enabled": true,
        "warmup_epochs": 2,
        "easy_epochs": 5,
        "hard_epochs": 10
    },
    "domain_adaptation": {
        "enabled": true,
        "use_mixup": true,
        "use_adversarial": true
    }
}'::jsonb;

-- Add domain info to products (if not exists)
ALTER TABLE products
ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN DEFAULT false;

ALTER TABLE products
ADD COLUMN IF NOT EXISTS domain_source TEXT;  -- 'render', 'photo', 'mixed'
```

### 5.2 API Endpoints

```python
# apps/api/src/api/v1/training.py (Additions)

@router.post("/runs/sota")
async def create_sota_training_run(
    request: SOTATrainingRequest,
    db = Depends(get_supabase),
):
    """Create a SOTA training run with advanced configuration."""

    # Validate triplet mining run if specified
    if request.triplet_mining_run_id:
        mining_run = await db.get_triplet_mining_run(request.triplet_mining_run_id)
        if not mining_run or mining_run["status"] != "completed":
            raise HTTPException(400, "Invalid or incomplete triplet mining run")

    # Create training run
    run_data = {
        "name": request.name,
        "description": request.description,
        "base_model_type": request.base_model_type,
        "data_source": request.data_source,
        "dataset_id": request.dataset_id,
        "training_config": request.training_config,
        "sota_config": request.sota_config.dict(),
        "triplet_mining_run_id": request.triplet_mining_run_id,
        "total_epochs": request.sota_config.num_epochs,
        "status": "pending",
    }

    # Create run record
    result = await db.create_training_run(run_data)

    return {"id": result["id"], "status": "pending"}
```

---

## Phase 6: Frontend Integration

### 6.1 Training Page Enhancements

```tsx
// apps/web/src/app/training/components/SOTAConfigPanel.tsx

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

interface SOTAConfig {
  loss: {
    loss_type: string;
    arcface_weight: number;
    triplet_weight: number;
    domain_weight: number;
    arcface_margin: number;
    triplet_margin: number;
  };
  sampling: {
    strategy: string;
    products_per_batch: number;
    samples_per_product: number;
    synthetic_ratio: number;
  };
  curriculum: {
    enabled: boolean;
    warmup_epochs: number;
    easy_epochs: number;
    hard_epochs: number;
    finetune_epochs: number;
  };
  domain_adaptation: {
    enabled: boolean;
    use_mixup: boolean;
    use_adversarial: boolean;
    cross_domain_triplet_ratio: number;
  };
}

export function SOTAConfigPanel({
  config,
  onChange,
  tripletMiningRuns,
  selectedMiningRun,
  onMiningRunChange,
}: {
  config: SOTAConfig;
  onChange: (config: SOTAConfig) => void;
  tripletMiningRuns: any[];
  selectedMiningRun: string | null;
  onMiningRunChange: (id: string | null) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>SOTA Training Configuration</CardTitle>
      </CardHeader>
      <CardContent>
        <Accordion type="multiple" defaultValue={["loss", "sampling"]}>
          {/* Loss Configuration */}
          <AccordionItem value="loss">
            <AccordionTrigger>Loss Functions</AccordionTrigger>
            <AccordionContent className="space-y-4">
              <div className="space-y-2">
                <Label>Loss Type</Label>
                <Select
                  value={config.loss.loss_type}
                  onValueChange={(v) => onChange({
                    ...config,
                    loss: { ...config.loss, loss_type: v }
                  })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="arcface">ArcFace Only</SelectItem>
                    <SelectItem value="triplet">Triplet Only</SelectItem>
                    <SelectItem value="combined">Combined (Recommended)</SelectItem>
                    <SelectItem value="domain_adaptive">Domain Adaptive</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {config.loss.loss_type === "combined" && (
                <>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>ArcFace Weight</Label>
                      <span className="text-sm">{config.loss.arcface_weight}</span>
                    </div>
                    <Slider
                      value={[config.loss.arcface_weight]}
                      onValueChange={([v]) => onChange({
                        ...config,
                        loss: { ...config.loss, arcface_weight: v }
                      })}
                      min={0}
                      max={2}
                      step={0.1}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Triplet Weight</Label>
                      <span className="text-sm">{config.loss.triplet_weight}</span>
                    </div>
                    <Slider
                      value={[config.loss.triplet_weight]}
                      onValueChange={([v]) => onChange({
                        ...config,
                        loss: { ...config.loss, triplet_weight: v }
                      })}
                      min={0}
                      max={2}
                      step={0.1}
                    />
                  </div>
                </>
              )}
            </AccordionContent>
          </AccordionItem>

          {/* Sampling Configuration */}
          <AccordionItem value="sampling">
            <AccordionTrigger>Batch Sampling</AccordionTrigger>
            <AccordionContent className="space-y-4">
              <div className="space-y-2">
                <Label>Sampling Strategy</Label>
                <Select
                  value={config.sampling.strategy}
                  onValueChange={(v) => onChange({
                    ...config,
                    sampling: { ...config.sampling, strategy: v }
                  })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="random">Random</SelectItem>
                    <SelectItem value="balanced">Class Balanced</SelectItem>
                    <SelectItem value="pk_sampler">P-K Sampler (Recommended)</SelectItem>
                    <SelectItem value="domain_balanced">Domain Balanced</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {config.sampling.strategy === "pk_sampler" && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Products per Batch (P)</Label>
                      <Slider
                        value={[config.sampling.products_per_batch]}
                        onValueChange={([v]) => onChange({
                          ...config,
                          sampling: { ...config.sampling, products_per_batch: v }
                        })}
                        min={4}
                        max={32}
                        step={1}
                      />
                      <span className="text-sm text-muted-foreground">
                        {config.sampling.products_per_batch}
                      </span>
                    </div>
                    <div className="space-y-2">
                      <Label>Samples per Product (K)</Label>
                      <Slider
                        value={[config.sampling.samples_per_product]}
                        onValueChange={([v]) => onChange({
                          ...config,
                          sampling: { ...config.sampling, samples_per_product: v }
                        })}
                        min={2}
                        max={8}
                        step={1}
                      />
                      <span className="text-sm text-muted-foreground">
                        {config.sampling.samples_per_product}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Batch size = P × K = {config.sampling.products_per_batch * config.sampling.samples_per_product}
                  </p>
                </>
              )}
            </AccordionContent>
          </AccordionItem>

          {/* Curriculum Learning */}
          <AccordionItem value="curriculum">
            <AccordionTrigger>Curriculum Learning</AccordionTrigger>
            <AccordionContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Enable Curriculum Learning</Label>
                <Switch
                  checked={config.curriculum.enabled}
                  onCheckedChange={(v) => onChange({
                    ...config,
                    curriculum: { ...config.curriculum, enabled: v }
                  })}
                />
              </div>

              {config.curriculum.enabled && (
                <div className="space-y-4 pt-2">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Warmup Epochs</Label>
                      <Slider
                        value={[config.curriculum.warmup_epochs]}
                        onValueChange={([v]) => onChange({
                          ...config,
                          curriculum: { ...config.curriculum, warmup_epochs: v }
                        })}
                        min={0}
                        max={5}
                        step={1}
                      />
                      <span className="text-sm">{config.curriculum.warmup_epochs}</span>
                    </div>
                    <div className="space-y-2">
                      <Label>Easy Phase Epochs</Label>
                      <Slider
                        value={[config.curriculum.easy_epochs]}
                        onValueChange={([v]) => onChange({
                          ...config,
                          curriculum: { ...config.curriculum, easy_epochs: v }
                        })}
                        min={1}
                        max={10}
                        step={1}
                      />
                      <span className="text-sm">{config.curriculum.easy_epochs}</span>
                    </div>
                    <div className="space-y-2">
                      <Label>Hard Phase Epochs</Label>
                      <Slider
                        value={[config.curriculum.hard_epochs]}
                        onValueChange={([v]) => onChange({
                          ...config,
                          curriculum: { ...config.curriculum, hard_epochs: v }
                        })}
                        min={1}
                        max={20}
                        step={1}
                      />
                      <span className="text-sm">{config.curriculum.hard_epochs}</span>
                    </div>
                    <div className="space-y-2">
                      <Label>Finetune Epochs</Label>
                      <Slider
                        value={[config.curriculum.finetune_epochs]}
                        onValueChange={([v]) => onChange({
                          ...config,
                          curriculum: { ...config.curriculum, finetune_epochs: v }
                        })}
                        min={0}
                        max={10}
                        step={1}
                      />
                      <span className="text-sm">{config.curriculum.finetune_epochs}</span>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Total epochs: {
                      config.curriculum.warmup_epochs +
                      config.curriculum.easy_epochs +
                      config.curriculum.hard_epochs +
                      config.curriculum.finetune_epochs
                    }
                  </p>
                </div>
              )}
            </AccordionContent>
          </AccordionItem>

          {/* Domain Adaptation */}
          <AccordionItem value="domain">
            <AccordionTrigger>Domain Adaptation</AccordionTrigger>
            <AccordionContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Enable Domain Adaptation</Label>
                <Switch
                  checked={config.domain_adaptation.enabled}
                  onCheckedChange={(v) => onChange({
                    ...config,
                    domain_adaptation: { ...config.domain_adaptation, enabled: v }
                  })}
                />
              </div>

              {config.domain_adaptation.enabled && (
                <>
                  <div className="flex items-center justify-between">
                    <Label>Use MixUp Augmentation</Label>
                    <Switch
                      checked={config.domain_adaptation.use_mixup}
                      onCheckedChange={(v) => onChange({
                        ...config,
                        domain_adaptation: { ...config.domain_adaptation, use_mixup: v }
                      })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>Use Adversarial Training</Label>
                    <Switch
                      checked={config.domain_adaptation.use_adversarial}
                      onCheckedChange={(v) => onChange({
                        ...config,
                        domain_adaptation: { ...config.domain_adaptation, use_adversarial: v }
                      })}
                    />
                  </div>
                </>
              )}
            </AccordionContent>
          </AccordionItem>

          {/* Triplet Mining */}
          <AccordionItem value="triplets">
            <AccordionTrigger>Pre-mined Triplets</AccordionTrigger>
            <AccordionContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Triplet Mining Run</Label>
                <Select
                  value={selectedMiningRun || "none"}
                  onValueChange={(v) => onMiningRunChange(v === "none" ? null : v)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Online mining (no pre-computed)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">Online Mining Only</SelectItem>
                    {tripletMiningRuns
                      .filter(r => r.status === "completed")
                      .map(run => (
                        <SelectItem key={run.id} value={run.id}>
                          {run.name} ({run.total_triplets} triplets)
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Pre-mined triplets provide better hard negatives for training
                </p>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
```

---

## Implementation Priority

### Phase 1: Core (Highest Priority)
1. ⬜ Database migrations for triplet mining
2. ⬜ Triplet mining API endpoints
3. ⬜ CombinedProductLoss implementation
4. ⬜ PKDomainSampler implementation

### Phase 2: Training Integration
1. ⬜ SOTATrainer implementation
2. ⬜ Curriculum learning logic
3. ⬜ LLRD optimizer setup
4. ⬜ Training config schema update

### Phase 3: Frontend
1. ⬜ Triplets page full implementation
2. ⬜ SOTA config panel for training page
3. ⬜ Triplet viewer component
4. ⬜ Mining statistics visualization

### Phase 4: Worker Integration
1. ⬜ Update RunPod worker to use SOTATrainer
2. ⬜ Add triplet loading from database
3. ⬜ Implement domain-aware data loading
4. ⬜ Add curriculum phase callbacks

### Phase 5: Extraction Integration (NEW)
1. ⬜ Add domain field to extraction payload
2. ⬜ Update extraction UI with domain selector
3. ⬜ Implement trained model registration endpoint
4. ⬜ Update API client with domain-aware methods
5. ⬜ Collection naming with domain suffix

---

## Model Compatibility

| Model | ArcFace | Triplet | Domain Adapt | Notes |
|-------|---------|---------|--------------|-------|
| DINOv2-Small | ✅ | ✅ | ✅ | 384-dim embeddings |
| DINOv2-Base | ✅ | ✅ | ✅ | 768-dim embeddings (default) |
| DINOv2-Large | ✅ | ✅ | ✅ | 1024-dim embeddings |
| DINOv3-Small | ✅ | ✅ | ✅ | 384-dim, newer architecture |
| DINOv3-Base | ✅ | ✅ | ✅ | 768-dim, recommended |
| DINOv3-Large | ✅ | ✅ | ✅ | 1024-dim, best quality |
| CLIP ViT-B/16 | ✅ | ✅ | ⚠️ | 512-dim, text-image pretrained |
| CLIP ViT-B/32 | ✅ | ✅ | ⚠️ | 512-dim, faster |
| CLIP ViT-L/14 | ✅ | ✅ | ⚠️ | 768-dim, highest quality |

**Note:** CLIP models may require different domain adaptation strategy due to text-image pretraining.

---

## Tiered Fallback System

For products with missing data:

```
Tier A: Full Triplet (Real + Synthetic + Hard Negative)
├── Anchor: Synthetic render
├── Positive: Real photo
└── Hard Negative: Similar real product

Tier B: Semi Triplet (Synthetic only + Hard Negative)
├── Anchor: Synthetic render (frame_0)
├── Positive: Synthetic render (frame_n)
└── Hard Negative: Similar synthetic product

Tier C: Synthetic Only (No negatives)
├── Anchor: Synthetic render
├── Positive: Another synthetic render
└── Random Negative: Any different product

Tier D: No Triplet (ArcFace only)
└── Single synthetic render → Classification only
```

---

## Phase 7: Embedding Extraction Integration

### 7.1 Domain-Aware Extraction

Triplet mining ve domain adaptation için embedding'lerde domain bilgisi olmalı.

```python
# apps/api/src/api/v1/embeddings.py (Additions)

class ProductExtractionConfig(BaseModel):
    """Enhanced extraction config with domain support."""

    # Existing fields
    frame_selection: str = "key_frames"  # first, key_frames, interval, all
    max_frames: int = 5
    include_first_frame: bool = True

    # NEW: Domain tracking
    mark_as_synthetic: bool = True  # Products from 3D renders
    domain_source: str = "render"  # render, photo, mixed

    # Collection config
    separate_collections: bool = True
    collection_prefix: str = "products"


async def extract_product_embeddings(
    product: dict,
    model: EmbeddingModel,
    config: ProductExtractionConfig,
) -> list[dict]:
    """Extract embeddings with domain info."""

    embeddings = []
    frames = select_frames(product, config)

    for i, frame_idx in enumerate(frames):
        # Get embedding
        image = load_frame(product, frame_idx)
        vector = model.encode(image)

        # Build payload with domain info
        payload = {
            "product_id": product["id"],
            "frame_index": frame_idx,
            "is_primary": i == 0,

            # Domain info for triplet mining
            "domain": "synthetic" if config.mark_as_synthetic else "real",
            "domain_source": config.domain_source,

            # Metadata
            "barcode": product.get("barcode"),
            "brand_name": product.get("brand_name"),
            "category": product.get("category"),
        }

        embeddings.append({
            "id": f"{product['id']}_{frame_idx}",
            "vector": vector.tolist(),
            "payload": payload,
        })

    return embeddings
```

### 7.2 Qdrant Payload Schema

```typescript
// Qdrant embedding payload structure
interface EmbeddingPayload {
  // Identity
  product_id: string;
  frame_index: number;
  is_primary: boolean;

  // Domain (CRITICAL for triplet mining)
  domain: "synthetic" | "real";
  domain_source: "render" | "photo" | "mixed";

  // Metadata
  barcode?: string;
  brand_name?: string;
  category?: string;

  // Training info (after extraction for training)
  training_run_id?: string;
  label_index?: number;  // Class index for this product
}
```

### 7.3 Extraction UI Updates

```tsx
// apps/web/src/app/embeddings/components/MatchingExtractionTab.tsx

// Add domain selection
<div className="space-y-2">
  <Label>Data Type</Label>
  <Select
    value={config.domain_source}
    onValueChange={(v) => setConfig({ ...config, domain_source: v })}
  >
    <SelectTrigger>
      <SelectValue />
    </SelectTrigger>
    <SelectContent>
      <SelectItem value="render">
        <div className="flex items-center gap-2">
          <Box className="h-4 w-4" />
          3D Renders (Synthetic)
        </div>
      </SelectItem>
      <SelectItem value="photo">
        <div className="flex items-center gap-2">
          <Camera className="h-4 w-4" />
          Real Photos
        </div>
      </SelectItem>
      <SelectItem value="mixed">
        <div className="flex items-center gap-2">
          <Shuffle className="h-4 w-4" />
          Mixed Sources
        </div>
      </SelectItem>
    </SelectContent>
  </Select>
  <p className="text-xs text-muted-foreground">
    Domain info is used for triplet mining and domain adaptation
  </p>
</div>
```

### 7.4 Trained Model → Extraction Flow

Eğitim tamamlandığında model extraction için kullanılabilir olmalı:

```python
# apps/api/src/api/v1/training.py

@router.post("/runs/{run_id}/register-model")
async def register_trained_model(
    run_id: str,
    request: RegisterModelRequest,
    db = Depends(get_supabase),
):
    """Register a trained model for extraction use."""

    # Get training run and best checkpoint
    run = await db.get_training_run(run_id)
    checkpoint = await db.get_best_checkpoint(run_id)

    if not checkpoint:
        raise HTTPException(400, "No checkpoint found")

    # Create embedding_models entry
    model_data = {
        "name": request.name or f"Fine-tuned {run['base_model_type']}",
        "model_type": "custom",
        "model_family": run["base_model_type"].split("-")[0],  # dinov2, dinov3, clip
        "embedding_dim": get_model_dim(run["base_model_type"]),
        "is_pretrained": False,
        "is_default": False,
        "base_model_id": await db.get_base_model_id(run["base_model_type"]),
        "config": {
            "checkpoint_url": checkpoint["checkpoint_url"],
            "training_run_id": run_id,
            "base_model_type": run["base_model_type"],
            "sota_config": run.get("sota_config"),
        },
        "product_collection": f"products_{request.collection_suffix}",
        "cutout_collection": f"cutouts_{request.collection_suffix}",
    }

    result = await db.create_embedding_model(model_data)

    # Update trained_models table
    await db.update_trained_model(
        checkpoint["id"],
        {"embedding_model_id": result["id"]}
    )

    return {"embedding_model_id": result["id"]}
```

### 7.5 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. EXTRACTION (Synthetic)                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   Products   │────▶│  Extraction  │────▶│   Qdrant     │            │
│  │  (3D Render) │     │   Worker     │     │ (domain:syn) │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                    │                     │
│  2. EXTRACTION (Real - if available)               │                     │
│  ┌──────────────┐     ┌──────────────┐            │                     │
│  │   Products   │────▶│  Extraction  │────────────┤                     │
│  │   (Photos)   │     │   Worker     │            │                     │
│  └──────────────┘     └──────────────┘            │                     │
│                                                    ▼                     │
│  3. TRIPLET MINING                         ┌──────────────┐            │
│  ┌──────────────┐     ┌──────────────┐     │   Qdrant     │            │
│  │   Mining     │◀────│   Mining     │◀────│ Collection   │            │
│  │   Results    │     │   Service    │     │              │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│         │                                                                │
│         ▼                                                                │
│  4. TRAINING                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   Triplets   │────▶│    SOTA      │────▶│   Trained    │            │
│  │  + Products  │     │   Trainer    │     │    Model     │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                    │                     │
│                                                    ▼                     │
│  5. MODEL REGISTRATION                     ┌──────────────┐            │
│  ┌──────────────┐     ┌──────────────┐     │  embedding   │            │
│  │   Register   │────▶│   Create     │────▶│   _models    │            │
│  │   Endpoint   │     │   Entry      │     │    table     │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                    │                     │
│                                                    ▼                     │
│  6. RE-EXTRACTION (with fine-tuned model)  ┌──────────────┐            │
│  ┌──────────────┐     ┌──────────────┐     │   New Qdrant │            │
│  │   Products   │────▶│  Extraction  │────▶│  Collection  │            │
│  │              │     │ (new model)  │     │              │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                    │                     │
│                                                    ▼                     │
│  7. MATCHING                               ┌──────────────┐            │
│  ┌──────────────┐     ┌──────────────┐     │   Better     │            │
│  │   Cutout     │────▶│   Matching   │────▶│   Results!   │            │
│  │   Image      │     │   Service    │     │              │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.6 Collection Naming Convention

| Purpose | Collection Name | Domain | Notes |
|---------|-----------------|--------|-------|
| Matching (Synthetic) | `products_{model}_synthetic` | synthetic | 3D renders |
| Matching (Real) | `products_{model}_real` | real | Photos |
| Matching (Mixed) | `products_{model}` | mixed | Both |
| Cutouts | `cutouts_{model}` | real | Customer uploads |
| Training | `training_{model}_{run_id}` | both | For specific run |

### 7.7 API Client Updates

```typescript
// apps/web/src/lib/api-client.ts (Additions)

export interface ExtractionConfig {
  // Existing
  frame_selection: "first" | "key_frames" | "interval" | "all";
  max_frames: number;
  include_first_frame: boolean;

  // NEW: Domain
  mark_as_synthetic: boolean;
  domain_source: "render" | "photo" | "mixed";

  // Collection
  separate_collections: boolean;
  collection_prefix: string;
}

export const apiClient = {
  // ... existing methods ...

  // Start extraction with domain info
  async startMatchingExtraction(config: {
    model_id: string;
    product_ids?: string[];
    dataset_id?: string;
    extraction_config: ExtractionConfig;
  }) {
    return this.post("/embeddings/extract/matching", config);
  },

  // Get collection with domain filter
  async getCollectionByDomain(
    collectionName: string,
    domain?: "synthetic" | "real"
  ) {
    const params = domain ? `?domain=${domain}` : "";
    return this.get(`/qdrant/collections/${collectionName}${params}`);
  },

  // Register trained model for extraction
  async registerTrainedModel(
    runId: string,
    name: string,
    collectionSuffix: string
  ) {
    return this.post(`/training/runs/${runId}/register-model`, {
      name,
      collection_suffix: collectionSuffix,
    });
  },
};
```

---

## Testing Checklist

### Triplet Mining
- [ ] Triplet mining produces valid triplets
- [ ] Cross-domain triplets identified correctly
- [ ] Difficulty classification (hard/semi-hard/easy) works

### Training
- [ ] Combined loss converges properly
- [ ] P-K sampler maintains class balance
- [ ] Curriculum phases transition correctly
- [ ] Domain adaptation improves real→synthetic transfer
- [ ] LLRD improves fine-tuning stability
- [ ] Checkpointing works across phases
- [ ] Metrics logged correctly to wandb

### Extraction Integration
- [ ] Domain field correctly set in extraction payload
- [ ] Synthetic vs Real filtering works in triplet mining
- [ ] Trained model registration creates embedding_models entry
- [ ] Fine-tuned model can be used for re-extraction
- [ ] Collection naming follows convention

---

## Appendix: Current System Verification (2026-01-17)

Bu bölüm mevcut sistemin durumunu doğrular ve gerçek eksiklikleri listeler.

### ✅ ZATEN MEVCUT (Plana Dahil Etmeye Gerek Yok)

| Feature | Location | Notes |
|---------|----------|-------|
| **Real Image Support** | `product_images.image_type` | `synthetic`, `real`, `augmented` enum |
| **Label Smoothing** | `workers/training/src/presets.py` | Tüm preset'lerde var |
| **Gradient Clipping** | `workers/training/src/presets.py` | `grad_clip: 1.0` ve `0.5` |
| **Freeze Backbone Methods** | `bb-models/base.py:93-117` | `freeze_backbone()`, `freeze_layers()` |
| **LLRD** | `workers/training/src/trainer.py` | Layer-wise LR decay implementasyonu |
| **ArcFace Loss** | `workers/training/src/losses.py` | Tam implementasyon |
| **GeM Pooling** | `bb-models/pooling.py` | Tam implementasyon |
| **Model Comparison API** | `api/v1/training.py:789-814` | `POST /training/models/compare` |
| **Model Comparison UI** | `training/page.tsx:292-325` | Checkbox + Compare button |
| **Augmentation System** | `workers/augmentation/src/augmentor.py` | `AugmentationConfig` dataclass |
| **Model Evaluations Table** | `migrations/011` | `model_evaluations` table |
| **Training Checkpoints** | `migrations/011` | `training_checkpoints` table |
| **Trained Models Registry** | `migrations/011` | `trained_models` table |

### ❌ GERÇEKTEN EKSİK (Bu Planda Ele Alınmalı)

| Feature | Priority | Status | Notes |
|---------|----------|--------|-------|
| **Evaluation Worker** | 🔴 HIGH | Phase 8 | Endpoint stubbed (TODO) |
| **Early Stopping Enforcement** | 🔴 HIGH | Phase 8 | Config var, loop'ta yok |
| **Triplet Mining System** | 🔴 HIGH | Phase 1 | Tamamen yok |
| **Triplet Loss** | 🔴 HIGH | Phase 2 | ArcFace var, Triplet yok |
| **Online Hard Mining** | 🔴 HIGH | Phase 2 | Yok |
| **P-K Batch Sampler** | 🔴 HIGH | Phase 3 | BalancedBatch var, P-K yok |
| **Curriculum Learning** | 🟡 MED | Phase 4 | Yok |
| **Domain Adversarial** | 🟡 MED | Phase 4 | Yok |
| **Feedback Loop** | 🟡 MED | Phase 9 | Opsiyonel |
| **ONNX Export** | 🟢 LOW | Future | bb-models'da yok |

### 📊 Tamamlanma Durumu

```
Mevcut Sistem: ████████████████░░░░ 82%
Bu Plan İle:   ████████████████████ 100%
```

---

## Phase 8: Missing System Components

### 8.1 Evaluation Worker Implementation

Mevcut durumda `/training/runs/{run_id}/evaluate` endpoint'i stubbed:

```python
# workers/evaluation/src/evaluator.py (YENİ WORKER)

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any

class ModelEvaluator:
    """Evaluate trained models on test set."""

    def __init__(self, model, test_dataset, device):
        self.model = model.to(device)
        self.test_dataset = test_dataset
        self.device = device

    def evaluate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run full evaluation."""
        self.model.eval()

        embeddings, labels, domains = self._extract_embeddings()

        results = {
            "overall_metrics": self._compute_retrieval_metrics(embeddings, labels),
        }

        if config.get("include_cross_domain", True):
            results.update(self._compute_cross_domain_metrics(embeddings, labels, domains))

        if config.get("include_per_category", True):
            results["per_category_metrics"] = self._compute_per_category_metrics(embeddings, labels)

        results["worst_product_ids"] = self._find_worst_products(embeddings, labels)
        results["most_confused_pairs"] = self._find_confused_pairs(embeddings, labels)

        return results

    def _extract_embeddings(self):
        loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        all_emb, all_labels, all_domains = [], [], []

        with torch.no_grad():
            for batch in loader:
                emb = self.model(batch["image"].to(self.device))
                all_emb.append(emb.cpu())
                all_labels.append(batch["label"])
                all_domains.append(batch.get("domain", torch.zeros_like(batch["label"])))

        return torch.cat(all_emb), torch.cat(all_labels), torch.cat(all_domains)

    def _compute_retrieval_metrics(self, embeddings, labels):
        sim = torch.mm(embeddings, embeddings.t())
        sim.fill_diagonal_(-float('inf'))

        metrics = {}
        for k in [1, 5, 10]:
            _, top_k = sim.topk(k, dim=1)
            correct = sum(1 for i in range(len(labels)) if labels[i] in labels[top_k[i]])
            metrics[f"recall@{k}"] = correct / len(labels)

        return metrics

    def _compute_cross_domain_metrics(self, embeddings, labels, domains):
        syn_mask = domains == 0
        real_mask = domains == 1
        results = {}

        if syn_mask.any() and real_mask.any():
            # Synthetic → Real
            sim = torch.mm(embeddings[syn_mask], embeddings[real_mask].t())
            _, top_1 = sim.topk(1, dim=1)
            syn_labels = labels[syn_mask]
            real_labels = labels[real_mask]
            correct = sum(1 for i in range(len(syn_labels)) if syn_labels[i] == real_labels[top_1[i, 0]])
            results["synthetic_to_real"] = {"recall@1": correct / len(syn_labels)}

            # Real → Synthetic
            sim = torch.mm(embeddings[real_mask], embeddings[syn_mask].t())
            _, top_1 = sim.topk(1, dim=1)
            correct = sum(1 for i in range(len(real_labels)) if real_labels[i] == syn_labels[top_1[i, 0]])
            results["real_to_synthetic"] = {"recall@1": correct / len(real_labels)}

        return results

    def _find_worst_products(self, embeddings, labels, top_k=20):
        sim = torch.mm(embeddings, embeddings.t())
        sim.fill_diagonal_(-float('inf'))

        product_recalls = {}
        for label in labels.unique():
            mask = labels == label
            if mask.sum() < 2:
                continue
            indices = torch.where(mask)[0]
            recalls = []
            for idx in indices:
                _, top_1 = sim[idx].topk(1)
                recalls.append(1 if labels[top_1[0]] == label else 0)
            product_recalls[label.item()] = np.mean(recalls)

        sorted_products = sorted(product_recalls.items(), key=lambda x: x[1])
        return [{"product_id": p, "recall@1": r} for p, r in sorted_products[:top_k]]

    def _find_confused_pairs(self, embeddings, labels, top_k=50):
        sim = torch.mm(embeddings, embeddings.t())
        sim.fill_diagonal_(-float('inf'))

        confusion = {}
        for i in range(len(labels)):
            _, top_1 = sim[i].topk(1)
            pred, true = labels[top_1[0]].item(), labels[i].item()
            if pred != true:
                pair = tuple(sorted([true, pred]))
                confusion[pair] = confusion.get(pair, 0) + 1

        sorted_pairs = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
        return [{"product_id_1": p[0], "product_id_2": p[1], "count": c} for p, c in sorted_pairs[:top_k]]
```

### 8.2 Early Stopping Implementation

```python
# workers/training/src/early_stopping.py (YENİ)

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (value > self.best_value + self.min_delta) if self.mode == "max" \
                   else (value < self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# Trainer'da kullanım:
# if config.get("early_stopping_patience", 0) > 0:
#     early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
#     if early_stopping(val_metrics["recall@1"]):
#         print(f"Early stopping triggered at epoch {epoch}")
#         break
```

---

## Phase 9: Feedback Loop (Opsiyonel)

### 9.1 Database Schema

```sql
-- Migration: 015_matching_feedback.sql

CREATE TABLE matching_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cutout_id UUID,
    cutout_image_url TEXT,
    predicted_product_id TEXT,
    predicted_similarity REAL,
    model_id UUID REFERENCES embedding_models(id),
    feedback_type TEXT CHECK (feedback_type IN ('correct', 'wrong', 'uncertain')),
    correct_product_id TEXT,
    user_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_wrong ON matching_feedback(feedback_type) WHERE feedback_type = 'wrong';
```

### 9.2 API Endpoint

```python
@router.post("/matching/feedback")
async def submit_feedback(request: FeedbackRequest, db = Depends(get_supabase)):
    await db.insert_matching_feedback(request.dict())
    return {"status": "recorded"}

@router.get("/matching/feedback/hard-examples")
async def get_hard_examples(model_id: str, db = Depends(get_supabase)):
    """Get wrong predictions as hard training examples."""
    wrong = await db.get_wrong_feedback(model_id)
    return [{"anchor": f["cutout_image_url"],
             "positive": f["correct_product_id"],
             "negative": f["predicted_product_id"]} for f in wrong]
```

---

## References

1. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
2. In Defense of the Triplet Loss for Person Re-Identification
3. Domain Adaptation for Object Recognition: An Unsupervised Approach
4. Curriculum Learning (Bengio et al.)
5. Layer-wise Learning Rate Decay for Fine-tuning Pretrained Models
