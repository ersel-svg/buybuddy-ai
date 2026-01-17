# BuyBuddy AI - E2E Test Plan

## Test Date: 2026-01-17
## Scope: Embedding Extraction, Triplet Mining, Model Training

---

## 1. Test Environment

### Local
- **API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **Database**: Supabase (cloud)
- **Vector DB**: Qdrant (cloud)

### GPU Pod (Compute)
- **Host**: root@213.173.102.137
- **Port**: 12070
- **SSH Key**: ~/.ssh/id_ed25519

---

## 2. Test Categories

### 2.1 Backend API Tests

| Test ID | Feature | Endpoint | Expected |
|---------|---------|----------|----------|
| API-001 | Health Check | GET /health | 200 OK |
| API-002 | Model Presets | GET /api/v1/training/presets | Model list |
| API-003 | Embedding Models | GET /api/v1/embeddings/models | Model list |
| API-004 | Qdrant Collections | GET /api/v1/embeddings/collections | Collection list |
| API-005 | Triplet Mining Runs | GET /api/v1/triplets/runs | Run list |
| API-006 | Training Runs | GET /api/v1/training/runs | Run list |
| API-007 | Label Stats | GET /api/v1/training/label-stats | Stats |
| API-008 | Products for Training | GET /api/v1/training/products | Product list |

### 2.2 Frontend Tests

| Test ID | Page | Component | Expected |
|---------|------|-----------|----------|
| FE-001 | /training | SOTAConfigPanel | Renders correctly |
| FE-002 | /training | Create Training Dialog | Form submits |
| FE-003 | /embeddings | Collections Tab | Shows collections |
| FE-004 | /embeddings | MatchingExtractionTab | Config works |
| FE-005 | /triplets | Triplet Mining Page | Loads data |
| FE-006 | /matching | Collection Selectors | Dropdowns work |

### 2.3 Worker Tests (GPU Pod)

| Test ID | Worker | Feature | Expected |
|---------|--------|---------|----------|
| WK-001 | embedding-extraction | DINOv2-base | Embeddings extracted |
| WK-002 | embedding-extraction | DINOv2-large | Embeddings extracted |
| WK-003 | embedding-extraction | DINOv3-base | Embeddings extracted |
| WK-004 | embedding-extraction | CLIP-ViT-L/14 | Embeddings extracted |
| WK-005 | training | Standard Trainer | Training completes |
| WK-006 | training | SOTAModelTrainer | SOTA features work |
| WK-007 | training | P-K Sampling | Batches created correctly |
| WK-008 | training | Combined Loss | All loss components |
| WK-009 | training | Recall@K | Metrics computed |
| WK-010 | training | Cross-domain Eval | Domain metrics |

### 2.4 Integration Tests

| Test ID | Flow | Steps | Expected |
|---------|------|-------|----------|
| INT-001 | Extract → Qdrant | Extract embeddings, store in Qdrant | Vectors searchable |
| INT-002 | Mine → Train | Mine triplets, use in training | Training uses triplets |
| INT-003 | Train → Evaluate | Train model, run evaluation | Metrics computed |
| INT-004 | SOTA Full Flow | Create SOTA training run from UI | End-to-end success |

---

## 3. Test Execution

### Phase 1: API Verification
```bash
# Health check
curl http://localhost:8000/health

# Model presets
curl http://localhost:8000/api/v1/training/presets

# Embedding collections
curl http://localhost:8000/api/v1/embeddings/collections
```

### Phase 2: Frontend Verification
```bash
# Build check
cd apps/web && npm run build

# Type check
npm run type-check
```

### Phase 3: Worker Tests (GPU Pod)
```bash
# Connect to pod
ssh root@213.173.102.137 -p 12070 -i ~/.ssh/id_ed25519

# Test embedding extraction
python -c "from extractor import get_extractor; e = get_extractor('dinov2-base'); print(e.embedding_dim)"

# Test training modules
python -c "from trainer import SOTAModelTrainer; print('SOTAModelTrainer OK')"
python -c "from losses import CombinedProductLoss; print('CombinedProductLoss OK')"
python -c "from samplers import PKDomainSampler; print('PKDomainSampler OK')"
```

---

## 4. Test Results

| Test ID | Status | Notes |
|---------|--------|-------|
| API-001 | | |
| API-002 | | |
| ... | | |

---

## 5. Issues Found

| Issue ID | Severity | Description | Resolution |
|----------|----------|-------------|------------|
| | | | |

---

## 6. Sign-off

- [ ] All API endpoints working
- [ ] Frontend builds without errors
- [ ] All model types extract embeddings
- [ ] SOTA training features functional
- [ ] Cross-domain evaluation working
- [ ] Triplet mining operational
