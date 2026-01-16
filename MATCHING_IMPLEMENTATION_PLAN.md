# Matching & Embedding System - Implementation Plan

## Overview
Bu plan, cutout sync, embedding extraction, matching UI ve export sisteminin implementasyonunu kapsar.

**Tarih**: 2025-01-14
**Durum**: In Progress

---

## Architecture Summary

- **Vector DB**: Qdrant (Cloud veya Self-hosted)
- **Collection**: Her model icin ayri collection (`embeddings_{model_id}`)
- **Worker**: Tek embedding worker (configurable model)
- **Export**: JSON, numpy, FAISS, Qdrant snapshot

---

## Phase 1: Foundation (Altyapi)

### 1.1 Database Migration
- [ ] `embedding_models` tablosu olustur
- [ ] `cutout_images` tablosu olustur
- [ ] `embedding_jobs` tablosu olustur
- [ ] `embedding_exports` tablosu olustur
- [ ] Gerekli index'leri ekle

**SQL:**
```sql
-- embedding_models
CREATE TABLE embedding_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  model_type TEXT NOT NULL,
  model_path TEXT,
  checkpoint_url TEXT,
  embedding_dim INTEGER NOT NULL,
  config JSONB DEFAULT '{}',
  qdrant_collection TEXT,
  qdrant_vector_count INTEGER DEFAULT 0,
  is_matching_active BOOLEAN DEFAULT false,
  training_job_id UUID REFERENCES jobs(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- cutout_images
CREATE TABLE cutout_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id INTEGER UNIQUE NOT NULL,
  image_url TEXT NOT NULL,
  predicted_upc TEXT,
  qdrant_point_id UUID,
  embedding_model_id UUID REFERENCES embedding_models(id),
  has_embedding BOOLEAN DEFAULT false,
  matched_product_id UUID REFERENCES products(id),
  match_similarity REAL,
  matched_by TEXT,
  matched_at TIMESTAMPTZ,
  synced_at TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cutout_external ON cutout_images(external_id);
CREATE INDEX idx_cutout_unmatched ON cutout_images(id) WHERE matched_product_id IS NULL;
CREATE INDEX idx_cutout_no_embedding ON cutout_images(id) WHERE has_embedding = false;
CREATE INDEX idx_cutout_predicted_upc ON cutout_images(predicted_upc);

-- embedding_jobs
CREATE TABLE embedding_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_type TEXT NOT NULL,
  model_id UUID REFERENCES embedding_models(id),
  target_type TEXT,
  product_ids UUID[],
  status TEXT DEFAULT 'pending',
  total_items INTEGER DEFAULT 0,
  processed_items INTEGER DEFAULT 0,
  failed_items INTEGER DEFAULT 0,
  runpod_job_id TEXT,
  config JSONB DEFAULT '{}',
  result JSONB DEFAULT '{}',
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ
);

-- embedding_exports
CREATE TABLE embedding_exports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id UUID REFERENCES embedding_models(id),
  format TEXT NOT NULL,
  filters JSONB DEFAULT '{}',
  status TEXT DEFAULT 'pending',
  download_url TEXT,
  file_size_bytes BIGINT,
  vector_count INTEGER,
  product_count INTEGER,
  exported_by TEXT,
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 1.2 Qdrant Setup
- [ ] Qdrant Cloud hesabi ac VEYA Docker compose'a ekle
- [ ] API key ve URL'i `.env`'e ekle
- [ ] `config.py`'ye Qdrant settings ekle

**Environment Variables:**
```
QDRANT_URL=https://xxx.qdrant.io
QDRANT_API_KEY=xxx
```

### 1.3 Qdrant Service
- [ ] `services/qdrant.py` olustur
  - [ ] `create_collection(name, vector_size)`
  - [ ] `delete_collection(name)`
  - [ ] `upsert_points(collection, points)`
  - [ ] `search(collection, vector, filter, limit)`
  - [ ] `scroll(collection, filter, with_vectors)`
  - [ ] `get_collection_info(name)`
  - [ ] `delete_points(collection, point_ids)`

### 1.4 BuyBuddy Cutout Integration
- [ ] `services/buybuddy.py`'ye `get_cutout_images()` ekle
  - [ ] Pagination destegi
  - [ ] Auth token kullanimi
  - [ ] Sorting parametreleri

---

## Phase 2: Cutout Sync

### 2.1 Backend - Cutout Endpoints
- [ ] `api/v1/cutouts.py` router olustur
  - [ ] `GET /cutouts` - Liste (paginated, filterable)
  - [ ] `POST /cutouts/sync` - BuyBuddy'den sync
  - [ ] `GET /cutouts/stats` - Istatistikler
  - [ ] `GET /cutouts/{id}` - Tek cutout detayi

### 2.2 Supabase Service
- [ ] `supabase.py`'ye cutout metodlari ekle
  - [ ] `sync_cutouts(cutouts)`
  - [ ] `get_cutouts(page, limit, filters)`
  - [ ] `get_cutout(id)`
  - [ ] `update_cutout(id, data)`
  - [ ] `get_cutout_stats()`

### 2.3 Frontend - Types
- [ ] `types/index.ts`'e cutout types ekle
  - [ ] `CutoutImage`
  - [ ] `CutoutSyncResponse`
  - [ ] `CutoutStats`

### 2.4 Frontend - API Client
- [ ] `api-client.ts`'e cutout metodlari ekle
  - [ ] `getCutouts()`
  - [ ] `syncCutouts()`
  - [ ] `getCutoutStats()`

---

## Phase 3: Embedding Models & Jobs

### 3.1 Backend - Model Endpoints
- [ ] `api/v1/embeddings.py` router olustur
  - [ ] `GET /embeddings/models`
  - [ ] `POST /embeddings/models`
  - [ ] `GET /embeddings/models/{id}`
  - [ ] `POST /embeddings/models/{id}/activate`
  - [ ] `DELETE /embeddings/models/{id}`

### 3.2 Backend - Job Endpoints
- [ ] Ayni router'da:
  - [ ] `POST /embeddings/extract`
  - [ ] `GET /embeddings/jobs`
  - [ ] `GET /embeddings/jobs/{id}`
  - [ ] `POST /embeddings/jobs/{id}/cancel`
  - [ ] `GET /embeddings/stats`

### 3.3 Supabase Service
- [ ] Model CRUD metodlari
- [ ] Job CRUD metodlari

### 3.4 Default Model Seed
- [ ] DINOv2-base modeli otomatik kaydet

---

## Phase 4: Embedding Worker

### 4.1 Worker Core
- [ ] `workers/embedding-worker/` klasor yapisi
  - [ ] `handler.py`
  - [ ] `models.py`
  - [ ] `extraction.py`
  - [ ] `qdrant_client.py`
  - [ ] `requirements.txt`
  - [ ] `Dockerfile`

### 4.2 Job Types
- [ ] `extract_cutouts`
- [ ] `extract_products`
- [ ] `extract_single_product`

### 4.3 RunPod Integration
- [ ] `services/runpod.py`'ye `EMBEDDING` endpoint type ekle
- [ ] Webhook handler ekle (`webhooks.py`)

### 4.4 Config
- [ ] `.env`'e `RUNPOD_EMBEDDING_ENDPOINT_ID` ekle

---

## Phase 5: Matching UI

### 5.1 Backend - Matching Endpoints
- [ ] `api/v1/matching.py` guncelle
  - [ ] `GET /matching/products/{id}/candidates`
  - [ ] `POST /matching/products/{id}/matches`

### 5.2 Frontend - Matching Page
- [ ] `app/matching/page.tsx` tamamen yenile
  - [ ] Sol panel: Product listesi
  - [ ] Orta panel: Reference gorsel + info
  - [ ] Sag panel: Candidates grid
  - [ ] Sync button
  - [ ] Extract embeddings button
  - [ ] Save matches button

### 5.3 Components
- [ ] `ProductListPanel.tsx`
- [ ] `ReferencePanel.tsx`
- [ ] `CandidatesGrid.tsx`
- [ ] `CandidateCard.tsx`
- [ ] `SyncCutoutsModal.tsx`
- [ ] `ExtractEmbeddingsModal.tsx`

---

## Phase 6: Export

### 6.1 Backend - Export Endpoints
- [ ] `api/v1/export.py` router olustur
  - [ ] `POST /export`
  - [ ] `GET /export`
  - [ ] `GET /export/{id}`
  - [ ] `GET /export/{id}/download`

### 6.2 Worker Export Job
- [ ] `export_embeddings` job type ekle
  - [ ] JSON format
  - [ ] Numpy format
  - [ ] FAISS format
  - [ ] Qdrant snapshot

### 6.3 S3 Storage
- [ ] Export dosyalari icin S3 bucket/path

### 6.4 Frontend - Export UI
- [ ] Embeddings sayfasina export section ekle

---

## Phase 7: Polish & Integration

### 7.1 Training Integration
- [ ] Training tamamlandiginda otomatik model kaydi
- [ ] Yeni model icin collection olusturma

### 7.2 Single Product Embedding
- [ ] Product detail sayfasina "Extract Embeddings" butonu
- [ ] Incremental Qdrant upsert

### 7.3 Stats Dashboard
- [ ] Embedding stats widget'i dashboard'a ekle

### 7.4 Error Handling & Logging
- [ ] Tum servislerde proper error handling
- [ ] Job failure recovery

---

## API Endpoints Summary

```
/api/v1/cutouts/
  GET    /                    - List cutouts
  POST   /sync                - Sync from BuyBuddy
  GET    /stats               - Statistics
  GET    /{id}                - Single cutout

/api/v1/embeddings/
  GET    /models              - List models
  POST   /models              - Register model
  GET    /models/{id}         - Model detail
  POST   /models/{id}/activate- Set active
  DELETE /models/{id}         - Delete model
  POST   /extract             - Start extraction
  GET    /jobs                - List jobs
  GET    /jobs/{id}           - Job status
  POST   /jobs/{id}/cancel    - Cancel job
  GET    /stats               - Statistics

/api/v1/matching/
  GET    /products            - Products for matching
  GET    /products/{id}       - Product detail
  GET    /products/{id}/candidates - Get candidates
  POST   /products/{id}/matches    - Save matches

/api/v1/export/
  POST   /                    - Create export
  GET    /                    - List exports
  GET    /{id}                - Export status
  GET    /{id}/download       - Download URL
```

---

## Progress Tracking

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| 1.1 Database Migration | Not Started | - | - |
| 1.2 Qdrant Setup | Not Started | - | - |
| 1.3 Qdrant Service | Not Started | - | - |
| 1.4 BuyBuddy Integration | Not Started | - | - |
| 2.1 Cutout Endpoints | Not Started | - | - |
| 2.2 Supabase Service | Not Started | - | - |
| 2.3 Frontend Types | Not Started | - | - |
| 2.4 API Client | Not Started | - | - |
| 3.x Embedding Models | Not Started | - | - |
| 4.x Embedding Worker | Not Started | - | - |
| 5.x Matching UI | Not Started | - | - |
| 6.x Export | Not Started | - | - |
| 7.x Polish | Not Started | - | - |

---

## Notes

- Cutout predicted_upc guvenilir degil, sadece hint olarak kullan
- Her sey product_id uzerinden (barcode secondary)
- Match edilen cutout -> product_images tablosuna real olarak eklenir
- Eski collection'lar kullanici silmediği surece kalir
