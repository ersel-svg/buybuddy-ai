# Buybuddy AI Platform - Project Plan

## 🎯 Hedef

Internal tool: Sahadan gelen ürün videolarını işleyip AI training data + product directory oluşturmak.

---

## Phase 1: Core Infrastructure (1-2 hafta)

### 1.1 Runpod Worker [ÖNCELİK: YÜKSEK]

**Amaç:** Pipeline'ı Docker container olarak Runpod Serverless'a deploy et.

**Tasks:**
- [ ] `worker/Dockerfile` oluştur
  - Base: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
  - SAM3 + dependencies install
  - ~8-10GB image size bekleniyor
  
- [ ] `worker/src/handler.py` - Runpod entrypoint
  - Input: `{video_url, barcode, video_id}`
  - Output: `{status, metadata, frame_count, frames_url}`
  
- [ ] `worker/src/pipeline.py` - Main pipeline class
  - Video download
  - Gemini extraction
  - SAM3 segmentation
  - Post-processing (518x518 frames)
  - Storage upload
  
- [ ] Local test with Docker
  ```bash
  docker build -t buybuddy-worker ./worker
  docker run --gpus all -e GEMINI_API_KEY=... buybuddy-worker
  ```
  
- [ ] Push to Docker Hub
- [ ] Create Runpod Serverless endpoint
- [ ] Test with real video

**Files:**
```
worker/
├── Dockerfile
├── requirements.txt
└── src/
    ├── handler.py
    ├── pipeline.py
    └── config.py
```

---

### 1.2 Supabase Setup [ÖNCELİK: YÜKSEK]

**Amaç:** Database + Storage + API

**Tasks:**
- [ ] Supabase project oluştur
- [ ] Database schema migrate et
  - `products` table
  - `jobs` table
- [ ] Storage bucket oluştur
  - `frames` bucket (public read)
- [ ] API keys al
- [ ] Row Level Security (RLS) kapat (internal tool)

**Schema:**
```sql
-- products table
CREATE TABLE products (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  barcode TEXT UNIQUE NOT NULL,
  video_id INTEGER,
  brand_name TEXT,
  sub_brand TEXT,
  product_name TEXT,
  variant_flavor TEXT,
  category TEXT,
  container_type TEXT,
  net_quantity TEXT,
  nutrition_facts JSONB,
  claims TEXT[],
  grounding_prompt TEXT,
  visibility_score INTEGER,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- jobs table
CREATE TABLE jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  barcode TEXT,
  video_url TEXT,
  video_id INTEGER,
  status TEXT DEFAULT 'pending',
  progress INTEGER DEFAULT 0,
  frame_count INTEGER,
  frames_path TEXT,
  error_message TEXT,
  runpod_job_id TEXT,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

---

### 1.3 NiceGUI Frontend [ÖNCELİK: ORTA]

**Amaç:** Basit internal UI

**Sayfalar:**

#### Dashboard (`/`)
- Toplam ürün sayısı
- Bekleyen job sayısı
- Son işlenen ürünler

#### Jobs (`/jobs`)
- Job listesi (status, barcode, created_at)
- "New Job" butonu → Buybuddy API'den ürün seç
- Job detay → progress, logs
- Retry failed jobs

#### Products (`/products`)
- Product listesi (filterable by status)
- Product detay:
  - Metadata görüntüle/düzenle
  - Frame gallery
  - Approve/Reject butonları
- Bulk operations

#### Sync (`/sync`)
- Buybuddy API'den ürün listesi çek
- İşlenmemiş ürünleri göster
- Batch process başlat

**Files:**
```
app/
├── main.py
├── pages/
│   ├── dashboard.py
│   ├── jobs.py
│   ├── products.py
│   └── sync.py
├── components/
│   ├── header.py
│   ├── sidebar.py
│   ├── video_player.py
│   ├── frame_gallery.py
│   └── metadata_editor.py
├── services/
│   ├── supabase.py
│   ├── runpod.py
│   └── buybuddy.py
└── requirements.txt
```

---

### 1.4 Integration [ÖNCELİK: YÜKSEK]

**Flow:**
```
UI → Supabase (create job) → Runpod Worker → Supabase (update job + save product)
                                    ↓
                              Storage (frames)
```

**Tasks:**
- [ ] UI'dan job başlat → Runpod'a request
- [ ] Runpod'dan webhook → Supabase update
- [ ] Realtime subscription → UI auto-refresh
- [ ] Error handling + retry logic

---

## Phase 2: Training Pipeline (2-3 hafta)

### 2.1 Domain Adaptation
- [ ] Real shelf image upload
- [ ] Synthetic frame ↔ Real image matching
- [ ] Matching score hesaplama

### 2.2 Augmentation Pipeline
- [ ] Augmentation config UI
  - Rotation range
  - Brightness/contrast
  - Background swap
  - Noise injection
- [ ] Batch augmentation job
- [ ] Preview augmented samples

### 2.3 Dataset Export
- [ ] COCO format export
- [ ] YOLO format export
- [ ] Train/val/test split
- [ ] Download as ZIP

### 2.4 Training Integration
- [ ] Training config
- [ ] Runpod training job
- [ ] Training progress monitoring
- [ ] Model artifact storage

---

## Phase 3: Embedding & Assignment (1-2 hafta)

### 3.1 Embedding Extraction
- [ ] Load trained model
- [ ] Extract embeddings from frames
- [ ] Average/aggregate per product

### 3.2 Vector Database
- [ ] Qdrant setup
- [ ] Bulk insert embeddings
- [ ] Similarity search API

### 3.3 Merchant Assignment
- [ ] Merchant product list import
- [ ] Auto-matching suggestions
- [ ] Manual assignment UI
- [ ] Export assignments

---

## 📅 Timeline

```
Week 1:
├── Day 1-2: Runpod Worker Dockerfile + handler.py
├── Day 3: Supabase setup + schema
├── Day 4-5: NiceGUI basic pages (jobs, products)

Week 2:
├── Day 1-2: Integration (UI → Runpod → Supabase)
├── Day 3-4: Error handling, retry, logging
├── Day 5: Testing, bug fixes

Week 3-4:
├── Training pipeline (Phase 2)

Week 5:
├── Embedding & Assignment (Phase 3)
```

---

## 🧪 Test Checklist

### Worker Tests
- [ ] Video download works
- [ ] Gemini extraction returns valid JSON
- [ ] SAM3 segments correctly
- [ ] Frames are 518x518
- [ ] Storage upload works
- [ ] Error handling works

### UI Tests
- [ ] Job list loads
- [ ] Can create new job
- [ ] Job status updates realtime
- [ ] Product list loads
- [ ] Can edit metadata
- [ ] Can approve/reject

### Integration Tests
- [ ] End-to-end: UI → Worker → Storage → UI
- [ ] Failed job retry
- [ ] Multiple concurrent jobs

---

## 🔧 Development Notes

### Local Development
```bash
# Worker (local Docker)
cd worker
docker build -t buybuddy-worker .
docker run --gpus all -p 8000:8000 \
  -e GEMINI_API_KEY=xxx \
  -e HF_TOKEN=xxx \
  buybuddy-worker

# UI (local)
cd app
pip install -r requirements.txt
python main.py
# Open http://localhost:8080
```

### Environment Variables

**Worker:**
```
GEMINI_API_KEY=your-gemini-api-key
HF_TOKEN=your-huggingface-token
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx
```

**UI:**
```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx
RUNPOD_API_KEY=xxx
RUNPOD_ENDPOINT_ID=xxx
BUYBUDDY_API_URL=https://api-legacy.buybuddy.co/api/v1
BUYBUDDY_USERNAME=your-username
BUYBUDDY_PASSWORD=your-password
```
