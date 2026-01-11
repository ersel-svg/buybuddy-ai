# Buybuddy AI Platform - Project Context

## 🎯 Proje Amacı

Sahadan gelen ürün videolarından:
1. **Product Directory** oluşturmak (Gemini ile metadata extraction)
2. **AI Training Data** oluşturmak (SAM3 ile segmentation → clean frames)
3. **Model Training** için veri hazırlamak (augmentation, domain adaptation)
4. **Embedding + Merchant Assignment** (trained model ile)

## 🏗️ Mimari Kararlar

### Stack
- **GPU Worker**: Runpod Serverless (Docker-based, pay-per-second)
- **Database**: Supabase (PostgreSQL + Storage + Realtime)
- **UI**: NiceGUI (Python-based, internal tool)
- **Deployment**: Runpod (worker), Vercel/Railway (UI)

### Neden Bu Seçimler?
- Runpod Serverless: SAM3 için GPU gerekli, cold start ~30s, sonra hızlı
- Supabase: Free tier yeterli, realtime updates, storage built-in
- NiceGUI: Ekipte Python deneyimi var, hızlı geliştirme

## 🔄 Pipeline Flow

```
1. Buybuddy API'den video URL al
2. Video'yu indir
3. Gemini Flash 2.5 → Metadata extraction (brand, product, nutrition, etc.)
4. SAM3 → Video segmentation (text prompt ile)
5. Post-process → 518x518 centered frames (siyah background)
6. Storage'a kaydet (frames + metadata)
7. UI'da göster, QA/approve workflow
```

## 📡 API Entegrasyonları

### Buybuddy API (Mevcut)
```
Base URL: https://api-legacy.buybuddy.co/api/v1

# Login
POST /user/sign_in
Body: {"user_name": "your-username", "password": "your-password"}
Response: {"passphrase": "..."}

# Get Token
POST /user/sign_in/token
Body: {"passphrase": "..."}
Response: {"token": "..."}

# Get Products
GET /ai/product
Headers: {"Authorization": "Bearer {token}"}
Response: {
  "data": [
    {
      "video": {
        "id": 1367260,
        "media_url": "https://bb-item-images.s3.amazonaws.com/xxx.mp4",
        "product": {
          "upc": "934455001100",
          "name": "",
          "sku": ""
        }
      },
      "processed": false
    }
  ],
  "total": 4290
}
```

### Gemini API
```python
import google.generativeai as genai
genai.configure(api_key="your-gemini-api-key")

# Video upload
video_file = genai.upload_file(path=video_path)

# Wait for processing
while video_file.state.name == "PROCESSING":
    time.sleep(2)
    video_file = genai.get_file(video_file.name)

# Generate
model = genai.GenerativeModel("gemini-2.0-flash-exp")
response = model.generate_content([video_file, prompt])
```

### SAM3 (Facebook Research)
```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()

# Start session
response = video_predictor.handle_request(
    request=dict(type="start_session", resource_path=str(video_path))
)
session_id = response["session_id"]

# Add text prompt (Gemini'den gelen grounding_prompt)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="Peace Tea Sno-Berry Can",  # grounding_prompt
    )
)

# Propagate through all frames
for item in video_predictor.propagate_in_video(
    session_id=session_id,
    propagation_direction="forward",
    start_frame_idx=0,
    max_frame_num_to_track=num_frames
):
    frame_idx = item.get('frame_index')
    outputs = item.get('outputs')
    # outputs['out_binary_masks']: (1, H, W) bool
    # outputs['out_boxes_xywh']: (1, 4) float
    # outputs['out_probs']: (1,) float

# Close session
video_predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
```

## 📊 Data Models

### Gemini Output Schema
```json
{
  "brand_info": {
    "brand_name": "Peace Tea",
    "sub_brand": "",
    "manufacturer_country": ""
  },
  "product_identity": {
    "product_name": "Sno-Berry",
    "variant_flavor": "Sno-Berry",
    "product_category": "beverages",
    "container_type": "can"
  },
  "specifications": {
    "net_quantity_text": "16 FL OZ (1 PT) 473 mL",
    "pack_configuration": {
      "type": "single_unit",
      "item_count": 1
    },
    "identifiers": {
      "barcode": "4900055775",
      "sku_model_code": null
    }
  },
  "marketing_and_claims": {
    "claims_list": ["Natural Flavors", "No Colors Added"],
    "marketing_description": null
  },
  "nutrition_facts": {
    "serving_size": "1 Can",
    "calories": 140.0,
    "total_fat": "0g",
    "protein": "0g",
    "carbohydrates": "37g",
    "sugar": "37g"
  },
  "visual_grounding": {
    "grounding_prompt": "Peace Tea Sno-Berry can"
  },
  "extraction_metadata": {
    "visibility_score": 100,
    "issues_detected": []
  }
}
```

### Database Schema (Supabase)
```sql
-- Products
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
  status TEXT DEFAULT 'pending', -- pending, approved, rejected
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Processing Jobs
CREATE TABLE jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  barcode TEXT,
  video_url TEXT,
  status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
  frame_count INTEGER,
  frames_path TEXT,
  error_message TEXT,
  runpod_job_id TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ
);
```

## 🔑 Credentials

```
# Buybuddy API
Username: (see .env)
Password: (see .env)

# HuggingFace (SAM3 model access)
Token: (see .env - HF_TOKEN)

# Gemini
API Key: (see .env - GEMINI_API_KEY)
```

## 📁 Proje Yapısı

```
buybuddy-ai/
├── worker/                    # Runpod Serverless Worker
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── handler.py         # Runpod entrypoint
│       ├── pipeline.py        # Main pipeline class
│       └── config.py
│
├── app/                       # NiceGUI Frontend
│   ├── main.py                # App entrypoint
│   ├── pages/
│   │   ├── dashboard.py       # Overview
│   │   ├── jobs.py            # Processing jobs
│   │   ├── products.py        # Product directory
│   │   └── training.py        # Training pipeline (Phase 2)
│   ├── components/
│   │   ├── video_player.py
│   │   ├── frame_gallery.py
│   │   └── metadata_editor.py
│   └── requirements.txt
│
├── shared/                    # Shared code
│   ├── models.py              # Pydantic schemas
│   ├── supabase_client.py
│   └── runpod_client.py
│
├── notebooks/                 # Reference notebooks
│   └── pipeline_prototype.ipynb
│
├── CONTEXT.md                 # This file
├── PROJECT_PLAN.md            # Roadmap
└── .cursorrules               # Cursor AI instructions
```

## 🖥️ Post-Processing Details

### Frame Processing (518x518)
```python
def center_on_canvas(frame, mask, target_size=518):
    # 1. Mask'tan bounding box bul
    # 2. Crop et
    # 3. Mask uygula (background = siyah)
    # 4. Aspect ratio koruyarak resize
    # 5. Siyah canvas üzerine ortala
    # Output: (518, 518, 3) RGB image
```

### Neden 518x518?
- DINOv2 optimal input size
- Embedding extraction için standart

## 🚀 Runpod Serverless

### Input Format
```json
{
  "input": {
    "video_url": "https://bb-item-images.s3.amazonaws.com/xxx.mp4",
    "barcode": "934455001100",
    "video_id": 1367260
  }
}
```

### Output Format
```json
{
  "status": "success",
  "barcode": "934455001100",
  "metadata": { ... },
  "frame_count": 177,
  "frames_url": "https://storage.supabase.co/..."
}
```

### Cold Start
- İlk request: ~30-60 saniye (model yükleme)
- Sonraki requestler: ~20-30 saniye (video işleme)

## 📋 Yapılacaklar Özeti

### Phase 1: Core Pipeline (Şu an)
- [x] Notebook prototype çalışıyor
- [ ] Runpod worker dockerize
- [ ] Supabase setup
- [ ] NiceGUI basic UI

### Phase 2: Training Pipeline
- [ ] Domain adaptation (real shelf images ile matching)
- [ ] Augmentation pipeline
- [ ] Dataset export (COCO/YOLO format)
- [ ] Model training integration

### Phase 3: Embedding & Assignment
- [ ] Trained model ile embedding extraction
- [ ] Qdrant vector database
- [ ] Merchant product assignment

## 🧪 Test Edilen Ürün

```
Barcode: 934455001100
Video ID: 1367260
Product: Peace Tea Sno-Berry Can
Frame Count: 177
Video Duration: ~6 seconds
```
