# Object Detection Training System - Kapsamlı QA Test Planı

## 📋 İçindekiler

1. [Sistem Genel Bakış](#1-sistem-genel-bakış)
2. [Test Ortamı Gereksinimleri](#2-test-ortamı-gereksinimleri)
3. [Backend API Testleri](#3-backend-api-testleri)
4. [Worker Testleri](#4-worker-testleri)
5. [Frontend UI Testleri](#5-frontend-ui-testleri)
6. [End-to-End Entegrasyon Testleri](#6-end-to-end-entegrasyon-testleri)
7. [Performans & Stress Testleri](#7-performans--stress-testleri)
8. [Augmentation Testleri](#8-augmentation-testleri)
9. [Error Handling & Recovery Testleri](#9-error-handling--recovery-testleri)
10. [UX & Kullanılabilirlik Testleri](#10-ux--kullanılabilirlik-testleri)
11. [Gerçek Dünya Senaryoları](#11-gerçek-dünya-senaryoları)
12. [Test Önceliklendirme](#12-test-önceliklendirme)

---

## 1. Sistem Genel Bakış

### Mimari Özet

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ Training Wizard  │  │ Training List    │  │ Training Detail│ │
│  │ /od/training/new │  │ /od/training     │  │ /od/training/id│ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘ │
└───────────┼─────────────────────┼────────────────────┼──────────┘
            │                     │                    │
            ▼                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND API (FastAPI)                      │
│  POST /od/training         GET /od/training        GET /{id}     │
│  POST /od/training/webhook GET /{id}/metrics       /{id}/logs    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │ RunPod Service │  │ Supabase Svc   │  │ OD Export Service  │ │
│  └───────┬────────┘  └───────┬────────┘  └─────────┬──────────┘ │
└──────────┼───────────────────┼─────────────────────┼────────────┘
           │                   │                     │
           ▼                   ▼                     ▼
┌──────────────────┐  ┌─────────────────┐  ┌───────────────────────┐
│   RunPod Worker  │  │   Supabase DB   │  │  Supabase Storage     │
│ ┌──────────────┐ │  │ od_training_runs│  │  datasets/            │
│ │  handler.py  │ │  │ od_trained_models│  │  models/              │
│ │  RT-DETR     │ │  │ od_datasets     │  │  checkpoints/         │
│ │  D-FINE      │ │  └─────────────────┘  └───────────────────────┘
│ │  SOTA Trainer│ │
│ └──────────────┘ │
└──────────────────┘
```

### Desteklenen Modeller

| Model | Boyutlar | Lisans | Özellikler |
|-------|----------|--------|------------|
| RT-DETR | s, m, l | Apache 2.0 | Real-Time Detection Transformer |
| D-FINE | s, m, l, x | Apache 2.0 | Dense Fine-grained Annotations |

### SOTA Özellikleri

- **EMA** (Exponential Moving Average): Model ağırlıklarının hareketli ortalaması
- **LLRD** (Layer-wise Learning Rate Decay): Katman bazlı öğrenme oranı azalması
- **Mixed Precision**: FP16 ile hızlandırılmış training
- **Warmup + Cosine Annealing**: Öğrenme oranı schedule'ı
- **40+ Augmentation**: Mosaic, MixUp, CopyPaste, geometric, color vb.

---

## 2. Test Ortamı Gereksinimleri

### RunPod Pod Kurulumu

```bash
# Pod Gereksinimleri
- GPU: RTX 3090 / A4000 / A5000 (min 24GB VRAM)
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB+ SSD

# Gerekli Env Variables
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"
export RUNPOD_API_KEY="your-runpod-key"
```

### Test Datasetleri

Sistemde mevcut olan gerçek datasetlerle test yapılacak:

| Dataset Tipi | Minimum Gereksinimler | Test Senaryosu |
|--------------|----------------------|----------------|
| Çok Küçük | 50-100 images, 1-3 classes | Heavy augmentation testi |
| Küçük | 200-500 images, 5-10 classes | SOTA preset testi |
| Orta | 1000-5000 images, 10-20 classes | Standard training |
| Büyük | 10000+ images, 20+ classes | Performance ve scaling |

### Test Konfigürasyonları

```python
# Hızlı Smoke Test Config
QUICK_TEST_CONFIG = {
    "epochs": 2,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "model_type": "rt-detr",
    "model_size": "s",
    "augmentation_preset": "light"
}

# Full Training Test Config
FULL_TEST_CONFIG = {
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "model_type": "d-fine",
    "model_size": "m",
    "augmentation_preset": "sota-v2",
    "use_ema": True,
    "use_llrd": True,
    "mixed_precision": True
}
```

---

## 3. Backend API Testleri

### 3.1 Training Run CRUD Operations

#### TC-API-001: Training Run Oluşturma (Başarılı)
```
Endpoint: POST /api/v1/od/training
Önkoşul: Geçerli dataset ve class mapping mevcut
Input:
{
  "name": "QA Test Training - TC-001",
  "description": "Smoke test for training creation",
  "dataset_id": "<valid_dataset_id>",
  "model_type": "rt-detr",
  "model_size": "s",
  "config": {
    "epochs": 2,
    "batch_size": 4,
    "learning_rate": 0.0001
  }
}
Beklenen: 201 Created, training_run_id döner
Doğrulama:
  - DB'de od_training_runs kaydı oluşturuldu
  - status: "pending" veya "preparing"
  - runpod_job_id atandı
```

#### TC-API-002: Training Run Oluşturma (Geçersiz Dataset)
```
Endpoint: POST /api/v1/od/training
Input: { "dataset_id": "non-existent-uuid", ... }
Beklenen: 404 Not Found
Doğrulama: Error message "Dataset not found" içerir
```

#### TC-API-003: Training Run Oluşturma (Geçersiz Model)
```
Endpoint: POST /api/v1/od/training
Input: { "model_type": "invalid-model", ... }
Beklenen: 422 Unprocessable Entity
Doğrulama: Validation error detaylı mesaj
```

#### TC-API-004: Training Run Listesi
```
Endpoint: GET /api/v1/od/training?limit=10&status=training
Beklenen: 200 OK, training run array
Doğrulama:
  - Pagination çalışıyor
  - Status filter uygulanıyor
  - created_at DESC sıralı
```

#### TC-API-005: Training Run Detayı
```
Endpoint: GET /api/v1/od/training/{training_id}
Beklenen: 200 OK, full training run object
Doğrulama:
  - metrics_history array
  - current_epoch doğru
  - best_map değerleri
```

#### TC-API-006: Training İptal
```
Endpoint: POST /api/v1/od/training/{training_id}/cancel
Önkoşul: Training status = "training"
Beklenen: 200 OK
Doğrulama:
  - status: "cancelled"
  - RunPod job cancelled
```

#### TC-API-007: Training Silme
```
Endpoint: DELETE /api/v1/od/training/{training_id}
Önkoşul: Training status != "training"
Beklenen: 200 OK
Doğrulama:
  - DB kaydı silindi
  - İlişkili model dosyaları temizlendi (optional)
```

### 3.2 Metrics & Logs API

#### TC-API-008: Training Metrics
```
Endpoint: GET /api/v1/od/training/{training_id}/metrics
Beklenen: 200 OK
Response:
{
  "metrics_history": [
    {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "map": 0.15, ...},
    {"epoch": 2, ...}
  ]
}
Doğrulama:
  - Her epoch için metrics mevcut
  - mAP, mAP@50, mAP@75 değerleri [0, 1] aralığında
```

#### TC-API-009: Training Logs
```
Endpoint: GET /api/v1/od/training/{training_id}/logs
Beklenen: 200 OK
Doğrulama:
  - Log entries time-ordered
  - Error logs varsa status=failed ile uyumlu
```

### 3.3 Webhook Tests

#### TC-API-010: Progress Webhook
```
Endpoint: POST /api/v1/od/training/webhook
Payload:
{
  "training_run_id": "<id>",
  "status": "training",
  "current_epoch": 5,
  "metrics": {"train_loss": 0.3, "val_loss": 0.25, "map": 0.35}
}
Beklenen: 200 OK
Doğrulama: DB güncellemesi yapıldı
```

#### TC-API-011: Completion Webhook
```
Endpoint: POST /api/v1/od/training/webhook
Payload:
{
  "training_run_id": "<id>",
  "status": "completed",
  "model_url": "https://storage.../model.pt",
  "best_metrics": {"map": 0.75, "map_50": 0.85}
}
Beklenen: 200 OK
Doğrulama:
  - status = "completed"
  - od_trained_models kaydı oluşturuldu
  - model_url kaydedildi
```

#### TC-API-012: Error Webhook
```
Endpoint: POST /api/v1/od/training/webhook
Payload:
{
  "training_run_id": "<id>",
  "status": "failed",
  "error_message": "CUDA out of memory",
  "error_traceback": "..."
}
Beklenen: 200 OK
Doğrulama:
  - status = "failed"
  - error_message kaydedildi
```

---

## 4. Worker Testleri

### 4.1 Handler Unit Tests

#### TC-WORKER-001: Job Input Validation
```python
# Test: Valid job input
job = {
    "input": {
        "training_run_id": "uuid",
        "dataset_url": "https://...",
        "model_type": "rt-detr",
        "model_size": "s",
        "config": {...}
    }
}
# Beklenen: handler başarıyla başlar, dataset download eder
```

#### TC-WORKER-002: Invalid Model Type
```python
job = {"input": {"model_type": "invalid"}}
# Beklenen: ValueError, early exit with error status
```

#### TC-WORKER-003: Dataset Download Failure
```python
job = {"input": {"dataset_url": "https://invalid-url.com/404.zip"}}
# Beklenen: Download error, status=failed, error message
```

### 4.2 Augmentation Config Conversion

#### TC-WORKER-004: Frontend Config Conversion
```python
frontend_config = {
    "mosaic": {"enabled": True, "probability": 0.5, "img_size": 640},
    "mixup": {"enabled": True, "probability": 0.3, "alpha": 8.0},
    "horizontal_flip": {"enabled": True, "probability": 0.5}
}
# Test convert_frontend_augmentation_config()
# Beklenen: Backend format with "prob" instead of "probability"
```

#### TC-WORKER-005: Legacy Alias Conversion
```python
frontend_config = {"copy_paste": {"enabled": True, "probability": 0.2}}
# Beklenen: "copypaste" key in output (legacy alias support)
```

#### TC-WORKER-006: All 56 Augmentations
```python
# Her augmentation tipi için config conversion testi
augmentations = [
    "mosaic", "mosaic9", "mixup", "cutmix", "copypaste",
    "horizontal_flip", "vertical_flip", "rotate90", "random_rotate",
    # ... tüm 56 augmentation
]
# Beklenen: Her biri doğru formata dönüştürülür
```

### 4.3 Model Training Tests

#### TC-WORKER-007: RT-DETR Small Training
```python
config = {
    "model_type": "rt-detr",
    "model_size": "s",
    "epochs": 2,
    "batch_size": 4
}
# Beklenen: Training başlar, 2 epoch tamamlanır, checkpoint kaydedilir
```

#### TC-WORKER-008: RT-DETR Medium Training
```python
config = {"model_type": "rt-detr", "model_size": "m", ...}
```

#### TC-WORKER-009: RT-DETR Large Training
```python
config = {"model_type": "rt-detr", "model_size": "l", ...}
```

#### TC-WORKER-010: D-FINE Small Training
```python
config = {"model_type": "d-fine", "model_size": "s", ...}
```

#### TC-WORKER-011: D-FINE Medium Training
```python
config = {"model_type": "d-fine", "model_size": "m", ...}
```

#### TC-WORKER-012: D-FINE Large Training
```python
config = {"model_type": "d-fine", "model_size": "l", ...}
```

#### TC-WORKER-013: D-FINE XLarge Training
```python
config = {"model_type": "d-fine", "model_size": "x", ...}
```

### 4.4 SOTA Features Tests

#### TC-WORKER-014: EMA Training
```python
config = {
    "use_ema": True,
    "ema_decay": 0.9999,
    ...
}
# Doğrulama:
#   - EMA weights ayrı hesaplanıyor
#   - Best model EMA weights ile kaydediliyor
```

#### TC-WORKER-015: LLRD Training
```python
config = {
    "use_llrd": True,
    "llrd_decay": 0.9,
    "head_lr_factor": 10.0,
    ...
}
# Doğrulama:
#   - Backbone layers düşük LR
#   - Head layers yüksek LR
```

#### TC-WORKER-016: Mixed Precision Training
```python
config = {"mixed_precision": True, ...}
# Doğrulama:
#   - FP16 forward pass
#   - FP32 master weights
#   - GradScaler kullanılıyor
```

#### TC-WORKER-017: Warmup + Cosine Scheduler
```python
config = {
    "warmup_epochs": 5,
    "scheduler": "cosine",
    ...
}
# Doğrulama:
#   - İlk 5 epoch LR artar
#   - Sonra cosine decay
```

#### TC-WORKER-018: Early Stopping
```python
config = {
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    ...
}
# Doğrulama: 10 epoch improvement yoksa durur
```

### 4.5 Evaluation Tests

#### TC-WORKER-019: COCO mAP Evaluation
```python
# Training sonrası val set üzerinde evaluation
# Doğrulama:
#   - mAP hesaplanıyor
#   - mAP@50, mAP@75 doğru
#   - mAP@small, mAP@medium, mAP@large
```

#### TC-WORKER-020: Checkpoint Saving
```python
# Her epoch sonunda checkpoint test
# Doğrulama:
#   - checkpoint_{epoch}.pt kaydedildi
#   - best_model.pt güncelleniyor
#   - Supabase storage upload başarılı
```

---

## 5. Frontend UI Testleri

### 5.1 Training Wizard Navigation

#### TC-UI-001: Wizard Step Navigation (Forward)
```
Adımlar:
1. /od/training/new aç
2. Dataset seç
3. "Next" butonuna tıkla
Beklenen: Step 2 (Preprocessing) açılır
Doğrulama: Stepper'da Step 1 checkmark gösterir
```

#### TC-UI-002: Wizard Step Navigation (Backward)
```
Adımlar:
1. Step 3'e kadar ilerle
2. "Back" butonuna tıkla
Beklenen: Step 2'ye döner, veriler korunur
```

#### TC-UI-003: Stepper Click Navigation
```
Adımlar:
1. Step 5'e kadar tamamla
2. Stepper'da Step 2'ye tıkla
Beklenen: Step 2 açılır, tamamlanmış step'lere dönülebilir
```

#### TC-UI-004: Invalid Forward Navigation
```
Adımlar:
1. Step 1'de dataset seçmeden "Next" tıkla
Beklenen:
  - Validation error gösterilir
  - İlerleme engellenir
  - Error alert görünür
```

### 5.2 Dataset Step Tests

#### TC-UI-005: Dataset Yükleme
```
Adımlar:
1. Wizard aç
2. Dataset dropdown'a tıkla
Beklenen:
  - Skeleton loader gösterilir (~300ms)
  - Datasetler listelenir
  - Boş state gösterilmez
```

#### TC-UI-006: Dataset Seçimi ve Stats
```
Adımlar:
1. Dataset seç
2. Stats card'ı kontrol et
Beklenen:
  - Total Images görünür
  - Annotation Coverage progress bar
  - Class sayısı badge
  - Dataset Size badge (Small/Medium/Large)
```

#### TC-UI-007: Split Slider Validasyonu
```
Adımlar:
1. Train split'i 40%'a çek
Beklenen:
  - Error: "Training split must be at least 50%"
  - Next butonu disabled
```

#### TC-UI-008: Split Sum Validasyonu
```
Adımlar:
1. Train: 60%, Val: 25%, Test: 25% ayarla
Beklenen:
  - Error: "Splits must sum to 100%"
  - Otomatik düzeltme veya error
```

### 5.3 Preprocessing Step Tests

#### TC-UI-009: Target Size Seçimi
```
Adımlar:
1. 640px butonuna tıkla
Beklenen: Buton highlighted, state güncellenir
```

#### TC-UI-010: Tiling Toggle
```
Adımlar:
1. "Enable Tiling" toggle'ı aç
Beklenen:
  - Tile size slider görünür
  - Tile overlap slider görünür
  - Min object area slider görünür
```

#### TC-UI-011: Large Image Warning
```
Adımlar:
1. Target size 1280px seç
Beklenen: "High GPU memory usage" warning görünür
```

### 5.4 Augmentation Step Tests

#### TC-UI-012: Offline Augmentation Toggle
```
Adımlar:
1. Offline Augmentation toggle'ı aç
Beklenen:
  - Multiplier selector görünür
  - Augmentation categories görünür
  - Resulting size hesaplanır
```

#### TC-UI-013: Augmentation Multiplier
```
Adımlar:
1. 1000 image dataset seç
2. Offline aug aç
3. 5x multiplier seç
Beklenen: "Resulting: ~5000 images" gösterilir
```

#### TC-UI-014: Online Augmentation Preset
```
Adımlar:
1. Online Augmentation step'e git
2. SOTA-v2 kartına tıkla
Beklenen:
  - SOTA-v2 kart highlighted
  - "Recommended" badge görünür
  - Feature list görünür
```

#### TC-UI-015: Augmentation Warnings
```
Adımlar:
1. 10,000 image dataset seç
2. 5x offline multiplier seç (= 50,000 images)
Beklenen: Warning: "Large dataset may require significant disk space"
```

### 5.5 Model Step Tests

#### TC-UI-016: Model Type Değişimi
```
Adımlar:
1. RT-DETR seç
2. Size "l" seç
3. D-FINE'a geç
Beklenen:
  - D-FINE'da size "x" seçeneği görünür
  - RT-DETR'da yoktu
```

#### TC-UI-017: RT-DETR Size Options
```
Beklenen: s, m, l boyutları mevcut (x yok)
```

#### TC-UI-018: D-FINE Size Options
```
Beklenen: s, m, l, x boyutları mevcut
```

#### TC-UI-019: VRAM Estimation
```
Adımlar: Her model/size kombinasyonu seç
Beklenen: VRAM estimation güncellenir (e.g., "~8GB VRAM")
```

#### TC-UI-020: Freeze Backbone Toggle
```
Adımlar:
1. "Freeze backbone" toggle'ı aç
2. Freeze epochs slider görünür
Beklenen: Slider [0, 50] aralığında çalışır
```

### 5.6 Hyperparameters Step Tests

#### TC-UI-021: Epochs Input
```
Test Cases:
- 0 epoch: Error "Epochs must be at least 1"
- 1 epoch: Valid
- 500 epoch: Valid
- 501 epoch: Warning "May lead to overfitting"
```

#### TC-UI-022: Batch Size Selection
```
Test Cases:
- 4: Valid (small GPU)
- 16: Valid (default)
- 32: Valid
- 64: Valid (large GPU)
- 128: Warning "High GPU memory"
```

#### TC-UI-023: Learning Rate Selection
```
Test Cases:
- 0.00001: Valid (fine-tuning)
- 0.0001: Valid (default)
- 0.001: Valid
- 0.01: Warning "May cause instability"
```

#### TC-UI-024: SOTA Feature Toggles
```
Her toggle için test:
- EMA: Toggle → emaDecay field görünür
- LLRD: Toggle → llrdDecay, headLrFactor görünür
- Mixed Precision: Toggle works independently
- Gradient Clipping: Slider [0, 5] çalışır
```

#### TC-UI-025: Early Stopping Config
```
Adımlar:
1. Enable early stopping
2. Patience slider: 5-50 aralığında
Beklenen: Patience > epochs/2 için warning
```

### 5.7 Review Step Tests

#### TC-UI-026: Summary Display
```
Doğrulama: Tüm önceki step seçimleri doğru gösteriliyor
- Dataset name
- Split percentages
- Preprocessing settings
- Augmentation preset
- Model type/size
- Hyperparameters
- SOTA features badges
```

#### TC-UI-027: Training Name Validation
```
Test Cases:
- Empty name: Error, Submit disabled
- 100 chars: Valid
- 101 chars: Error "Name too long"
```

#### TC-UI-028: Training Time Estimation
```
Adımlar:
1. 1000 images, 100 epochs, batch 16, RT-DETR-L
Beklenen: "Estimated time: ~2h 30m" (yaklaşık)
```

### 5.8 Form Submission Tests

#### TC-UI-029: Successful Submission
```
Adımlar:
1. Tüm step'leri valid şekilde doldur
2. "Start Training" butonuna tıkla
Beklenen:
  - Button spinner gösterir
  - "Starting Training..." text
  - Success toast
  - Redirect to /od/training/{id}
```

#### TC-UI-030: Submission Error Handling
```
Adımlar:
1. Network offline simüle et
2. Submit
Beklenen:
  - Error toast gösterilir
  - Button re-enabled
  - Spinner durur
```

#### TC-UI-031: Double Submit Prevention
```
Adımlar:
1. Submit butonuna hızlıca 2 kez tıkla
Beklenen: Sadece 1 request gönderilir
```

### 5.9 Smart Defaults Tests

#### TC-UI-032: Smart Recommendations Display
```
Adımlar:
1. Küçük dataset (<500 images) seç
Beklenen:
  - "AI-Generated" badge
  - "Small dataset detected" analysis
  - Heavy augmentation recommended
  - Smaller model recommended
```

#### TC-UI-033: Apply Recommendations
```
Adımlar:
1. "Apply Recommendations" butonuna tıkla
Beklenen:
  - Tüm step'ler önerilen değerlerle doldu
  - Toast: "Smart defaults applied!"
  - Buton disabled: "Applied ✓"
```

### 5.10 Training Detail Page Tests

#### TC-UI-034: Progress Display
```
URL: /od/training/{training_id}
Doğrulama:
  - Current epoch / Total epochs
  - Progress bar
  - Current metrics
  - Loss chart
  - mAP chart
```

#### TC-UI-035: Real-time Updates
```
Adımlar:
1. Training detail page aç
2. 30 saniye bekle
Beklenen: Metrics her 5-10 saniyede güncellenir
```

#### TC-UI-036: Cancel Training
```
Adımlar:
1. Running training detail page
2. Cancel butonuna tıkla
3. Confirm dialog'da "Yes" tıkla
Beklenen:
  - Status: "Cancelled"
  - RunPod job cancelled
```

#### TC-UI-037: Completed Training View
```
Doğrulama:
  - Final metrics görünür
  - Best epoch highlighted
  - Download model butonu aktif
  - Full metrics history tablosu
```

### 5.11 Responsive Design Tests

#### TC-UI-038: Desktop Layout (1920x1080)
```
Doğrulama:
  - Wizard content 3/4 width
  - Smart recommendations sidebar 1/4 width
  - Full stepper visible
```

#### TC-UI-039: Tablet Layout (768x1024)
```
Doğrulama:
  - 2-column layout
  - Stacked components
  - Touch-friendly buttons
```

#### TC-UI-040: Mobile Layout (375x812)
```
Doğrulama:
  - Single column
  - Compact stepper "Step X of 7"
  - Full-width inputs
  - Thumb-friendly controls
```

---

## 6. End-to-End Entegrasyon Testleri

### 6.1 Happy Path: Full Training Cycle

#### TC-E2E-001: Complete Training Flow (RT-DETR-S, 5 epochs)
```
Süre: ~15-20 dakika

Adımlar:
1. Frontend: Wizard aç
2. Dataset: Küçük dataset seç (100-200 images)
3. Preprocessing: 640px, no tiling
4. Offline Aug: Disabled
5. Online Aug: Light preset
6. Model: RT-DETR-S
7. Hyperparams:
   - epochs: 5
   - batch_size: 8
   - learning_rate: 0.0001
   - EMA: enabled
   - Mixed precision: enabled
8. Review: Name gir, submit
9. Detail page: Progress takip et
10. Completion: Model download et

Doğrulama Noktaları:
□ Training başladı (status: training)
□ Epoch 1 metrics alındı
□ Epoch 5 tamamlandı
□ Status: completed
□ Model URL mevcut
□ od_trained_models kaydı oluştu
□ Best mAP > 0 (training çalıştı)
```

#### TC-E2E-002: Complete Training Flow (D-FINE-M, SOTA-v2, 10 epochs)
```
Süre: ~45-60 dakika

Adımlar:
1. Medium dataset seç (500-1000 images)
2. SOTA-v2 augmentation preset
3. D-FINE-M model
4. epochs: 10
5. Full SOTA features enabled

Doğrulama Noktaları:
□ Mosaic augmentation uygulanıyor
□ MixUp augmentation uygulanıyor
□ EMA weights ayrı kaydediliyor
□ LLRD çalışıyor (layer-wise LR)
□ mAP her epoch artıyor (genel trend)
□ Final mAP > initial mAP
```

#### TC-E2E-003: Large Dataset Training (D-FINE-L, 50 epochs)
```
Süre: ~4-8 saat

Adımlar:
1. Büyük dataset seç (5000+ images)
2. Medium augmentation preset
3. D-FINE-L model
4. epochs: 50, early_stopping: 15
5. Mixed precision enabled

Doğrulama Noktaları:
□ Dataset download < 5 dakika
□ Memory usage stable
□ No OOM errors
□ Checkpoint her 10 epoch kaydedildi
□ Early stopping tetiklendi veya 50 epoch tamamlandı
□ Best model saved correctly
```

### 6.2 Error Recovery Scenarios

#### TC-E2E-004: Training Cancellation & State
```
Adımlar:
1. Training başlat
2. Epoch 3'te cancel et
3. Training list'e dön

Doğrulama:
□ Status: cancelled
□ Metrics epoch 3'e kadar mevcut
□ RunPod job cancelled
□ Partial checkpoint mevcut (optional)
```

#### TC-E2E-005: Network Interruption Recovery
```
Adımlar:
1. Training başlat
2. Network connection kes (simüle et)
3. 30 saniye bekle
4. Network geri aç

Doğrulama:
□ Worker direct Supabase writes kullanıyor
□ Status güncellemeleri devam ediyor
□ Webhook dependency yok
```

#### TC-E2E-006: Invalid Dataset Handling
```
Adımlar:
1. Boş veya bozuk dataset ile training başlat

Doğrulama:
□ Anlamlı error message
□ Status: failed
□ error_traceback kaydedildi
□ Kullanıcı bilgilendirildi
```

### 6.3 Multi-Training Concurrent Tests

#### TC-E2E-007: Parallel Training Runs
```
Adımlar:
1. 2-3 training run aynı anda başlat
2. Tüm run'ları monitör et

Doğrulama:
□ Tüm run'lar bağımsız çalışıyor
□ RunPod queue yönetimi çalışıyor
□ Resource conflicts yok
□ Her run kendi progress'ini gösteriyor
```

---

## 7. Performans & Stress Testleri

### 7.1 Dataset Processing Performance

#### TC-PERF-001: Small Dataset Export Time
```
Dataset: 500 images
Beklenen: < 30 saniye export time
```

#### TC-PERF-002: Large Dataset Export Time
```
Dataset: 10,000 images
Beklenen: < 5 dakika export time
```

#### TC-PERF-003: Dataset Download Speed
```
ZIP Size: 1GB
Beklenen: < 2 dakika download (100Mbps+ connection)
```

### 7.2 Training Speed Benchmarks

#### TC-PERF-004: Training Speed - RT-DETR-S
```
GPU: RTX 3090
Dataset: 1000 images
Batch: 16
Beklenen: ~0.05s/step
```

#### TC-PERF-005: Training Speed - D-FINE-L
```
GPU: RTX 3090
Dataset: 1000 images
Batch: 8
Beklenen: ~0.15s/step
```

### 7.3 Memory Usage

#### TC-PERF-006: VRAM Usage - RT-DETR-S
```
Batch: 16, Image: 640x640
Beklenen: < 8GB VRAM
```

#### TC-PERF-007: VRAM Usage - D-FINE-X
```
Batch: 4, Image: 640x640
Beklenen: < 24GB VRAM
```

#### TC-PERF-008: OOM Recovery
```
Adımlar:
1. Batch size 64 ile training başlat
2. OOM bekleniyor

Doğrulama:
□ Error caught gracefully
□ Error message: "CUDA out of memory"
□ Suggestion: "Reduce batch size"
```

### 7.4 API Response Times

#### TC-PERF-009: Training List Response
```
Endpoint: GET /od/training?limit=50
Beklenen: < 500ms
```

#### TC-PERF-010: Training Detail Response
```
Endpoint: GET /od/training/{id}
Beklenen: < 200ms
```

#### TC-PERF-011: Metrics History Response
```
Endpoint: GET /od/training/{id}/metrics
50 epochs
Beklenen: < 300ms
```

---

## 8. Augmentation Testleri

### 8.1 Preset Tests

#### TC-AUG-001: SOTA-v2 Preset Loading
```python
preset = get_augmentation_preset("sota-v2")
assert preset.mosaic.enabled == True
assert preset.mosaic.prob == 0.5
assert preset.mixup.enabled == True
assert preset.copypaste.enabled == True
```

#### TC-AUG-002: Heavy Preset (20+ Augmentations)
```python
preset = get_augmentation_preset("heavy")
enabled_count = sum(1 for aug in preset if aug.enabled)
assert enabled_count >= 20
```

#### TC-AUG-003: None Preset (No Augmentations)
```python
preset = get_augmentation_preset("none")
enabled_count = sum(1 for aug in preset if aug.enabled)
assert enabled_count == 0
```

### 8.2 Multi-Image Augmentation Tests

#### TC-AUG-004: Mosaic Augmentation
```python
# 4 image'ı 2x2 grid'e birleştir
# BBox'lar doğru transform edilmeli
# Minimum bbox size filtering çalışmalı
```

#### TC-AUG-005: Mosaic-9 Augmentation
```python
# 9 image'ı 3x3 grid'e birleştir
```

#### TC-AUG-006: MixUp Augmentation
```python
# 2 image'ı alpha-blend et
# Labels interpolate edilmeli
```

#### TC-AUG-007: CopyPaste Augmentation
```python
# Bir image'dan objects kopyala
# Başka image'a yapıştır
# BBox'lar doğru eklenmeli
```

### 8.3 Geometric Augmentation Tests

#### TC-AUG-008: Horizontal Flip with BBox
```python
# Image flip edildiğinde bbox x koordinatları flip edilmeli
# x_new = image_width - x_old - width
```

#### TC-AUG-009: Rotation with BBox
```python
# 15° rotation
# BBox corners rotate edilip yeni axis-aligned bbox hesaplanmalı
```

#### TC-AUG-010: Scale with BBox
```python
# 0.8-1.2x scale
# BBox scale factor ile çarpılmalı
```

### 8.4 Edge Case Tests

#### TC-AUG-011: Empty Image (No Objects)
```python
# 0 bbox'lı image üzerinde augmentation
# Pipeline crash etmemeli
```

#### TC-AUG-012: Single Small Object
```python
# 5x5 pixel bbox
# min_bbox_size filtering sonrası kaybolabilir
# Uygun warning/handling
```

#### TC-AUG-013: Image Size Mismatch
```python
# MixUp için farklı boyutlu images
# Auto-resize veya padding uygulanmalı
```

#### TC-AUG-014: Invalid Probability
```python
# prob = 1.5 veya prob = -0.5
# Validation error veya clamp to [0, 1]
```

---

## 9. Error Handling & Recovery Testleri

### 9.1 API Error Handling

#### TC-ERR-001: Invalid JSON Body
```
POST /od/training
Body: "invalid json"
Beklenen: 422 Unprocessable Entity
```

#### TC-ERR-002: Missing Required Fields
```
POST /od/training
Body: {"name": "test"}  # missing dataset_id, model_type
Beklenen: 422 with detailed validation errors
```

#### TC-ERR-003: Database Connection Error
```
Simülasyon: Supabase connection timeout
Beklenen: 503 Service Unavailable
```

#### TC-ERR-004: RunPod API Error
```
Simülasyon: RunPod endpoint down
Beklenen:
- Training created with status "failed"
- Error message: "Failed to submit to RunPod"
```

### 9.2 Worker Error Handling

#### TC-ERR-005: Dataset Corrupted
```
Input: Bozuk ZIP file
Beklenen:
- status: "failed"
- error_message: "Failed to extract dataset"
```

#### TC-ERR-006: Model Loading Failure
```
Input: Invalid model_size
Beklenen:
- status: "failed"
- error_message: "Unsupported model configuration"
```

#### TC-ERR-007: Training Exception
```
Simülasyon: Division by zero in loss calculation
Beklenen:
- status: "failed"
- error_traceback: Full traceback
- Partial checkpoint saved (if available)
```

#### TC-ERR-008: Storage Upload Failure
```
Simülasyon: Supabase storage quota exceeded
Beklenen:
- Retry logic (3 attempts)
- Error message with retry count
- Training marked as failed after retries
```

### 9.3 Frontend Error Handling

#### TC-ERR-009: API Timeout
```
Simülasyon: API response > 30 seconds
Beklenen:
- Error toast shown
- Form re-enabled
- User can retry
```

#### TC-ERR-010: Network Offline
```
Simülasyon: navigator.onLine = false
Beklenen:
- Appropriate offline message
- Retry when online
```

#### TC-ERR-011: Invalid API Response
```
Simülasyon: API returns malformed JSON
Beklenen:
- Error caught
- User-friendly message
- No crash
```

---

## 10. UX & Kullanılabilirlik Testleri

### 10.1 Loading States

#### TC-UX-001: Dataset List Loading
```
Doğrulama:
- Skeleton loader görünür
- Loading süresi < 3 saniye
- No flash of empty content
```

#### TC-UX-002: Stats Card Loading
```
Doğrulama:
- Multiple skeleton bars
- Loading indicator
- Smooth transition to data
```

#### TC-UX-003: Submit Button Loading
```
Doğrulama:
- Spinner animation
- "Starting Training..." text
- Disabled state (no interaction)
```

### 10.2 Feedback & Notifications

#### TC-UX-004: Success Toast
```
Actions: Submit successful training
Beklenen:
- Green success toast
- "Training started!" message
- Auto-dismiss after 5s
```

#### TC-UX-005: Error Toast
```
Actions: Submit fails
Beklenen:
- Red error toast
- Detailed error message
- Dismissible by user
```

#### TC-UX-006: Warning Display
```
Actions: Select large batch size
Beklenen:
- Yellow warning inline
- Helpful suggestion text
- Doesn't block action
```

### 10.3 Form Usability

#### TC-UX-007: Input Validation Feedback
```
Doğrulama:
- Real-time validation
- Clear error messages
- Error clears when fixed
```

#### TC-UX-008: Step Completion Feedback
```
Doğrulama:
- Checkmark on completed step
- Visual progress indicator
- Clear current step highlight
```

#### TC-UX-009: Keyboard Navigation
```
Test:
- Tab through all inputs
- Enter submits current step
- Escape closes dialogs
```

### 10.4 Accessibility

#### TC-UX-010: Screen Reader Compatibility
```
Tools: NVDA, VoiceOver
Doğrulama:
- All inputs have labels
- Error messages announced
- Progress updates accessible
```

#### TC-UX-011: Color Contrast
```
Tools: aXe, Lighthouse
Doğrulama:
- WCAG AA compliance
- Errors visible to color-blind users
```

---

## 11. Gerçek Dünya Senaryoları

### 11.1 Retail Product Detection

#### TC-RW-001: Ürün Raf Tespiti
```
Dataset: Mağaza raf görüntüleri
Classlar: product, shelf, price_tag
Hedef mAP: > 0.70

Config:
- Model: D-FINE-M
- Epochs: 100
- Augmentation: SOTA-v2
- EMA + LLRD enabled
```

#### TC-RW-002: Barkod Alanı Tespiti
```
Dataset: Ürün görüntüleri
Classlar: barcode_area
Hedef mAP: > 0.85

Config:
- Model: RT-DETR-L
- High precision needed
- Light augmentation (barkod distortion istenmiyor)
```

### 11.2 Edge Cases in Real Data

#### TC-RW-003: Düşük Kaliteli Görüntüler
```
Dataset: Bulanık, düşük ışık görüntüleri
Doğrulama:
- Model degrade gracefully
- Augmentation helps (blur, noise)
```

#### TC-RW-004: Class Imbalance
```
Dataset: 1000 "product", 50 "defect"
Doğrulama:
- Class weights applied
- Minority class mAP reported
- Balanced sampling option
```

#### TC-RW-005: Overlapping Objects
```
Dataset: Yoğun raf görüntüleri
Doğrulama:
- NMS threshold tuning
- Crowded scene handling
```

### 11.3 Production Simulation

#### TC-RW-006: 24 Saat Continuous Training
```
Setup: 3 training run back-to-back
Doğrulama:
- Memory leaks yok
- Worker stable
- All runs complete
```

#### TC-RW-007: Peak Load Simulation
```
Setup: 5 concurrent training requests
Doğrulama:
- RunPod queue handles load
- No timeouts
- All trainings eventually start
```

---

## 12. Test Önceliklendirme

### Phase 1: Critical Path (Day 1-2)
```
[P0] TC-API-001: Training Run Oluşturma
[P0] TC-WORKER-007: RT-DETR-S Training (smoke test)
[P0] TC-UI-029: Successful Submission
[P0] TC-E2E-001: Complete Training Flow (5 epochs)
[P0] TC-ERR-007: Training Exception Handling
```

### Phase 2: Core Features (Day 3-5)
```
[P1] TC-WORKER-010 to 013: All Model Variants
[P1] TC-WORKER-014 to 018: SOTA Features
[P1] TC-AUG-001 to 007: Augmentation Presets
[P1] TC-UI-001 to 040: Full UI Coverage
[P1] TC-E2E-002: SOTA-v2 Training
```

### Phase 3: Edge Cases & Performance (Day 6-7)
```
[P2] TC-PERF-001 to 011: All Performance Tests
[P2] TC-ERR-001 to 011: All Error Handling
[P2] TC-AUG-008 to 014: Edge Case Augmentations
[P2] TC-E2E-003: Large Dataset Training
```

### Phase 4: Real World & Polish (Day 8-10)
```
[P3] TC-RW-001 to 007: Real World Scenarios
[P3] TC-UX-001 to 011: UX Polish
[P3] TC-E2E-007: Concurrent Training
```

---

## Appendix A: Test Execution Commands

### Backend API Tests
```bash
cd apps/api
pytest tests/ -v --tb=short
```

### Worker Tests
```bash
cd workers/od-training
pytest tests/ -v --tb=short
```

### E2E Tests (Manual)
```bash
# Start API locally
cd apps/api && uvicorn src.main:app --reload

# Start Frontend
cd apps/web && npm run dev

# Manual test execution via browser
```

### Performance Tests
```bash
# Using locust or k6
locust -f tests/perf/locustfile.py
```

---

## Appendix B: Test Data Requirements

### Minimum Test Datasets

| Dataset | Images | Classes | Annotations | Purpose |
|---------|--------|---------|-------------|---------|
| qa_smoke_test | 50 | 3 | 200 | Quick smoke tests |
| qa_small | 200 | 5 | 1000 | Unit tests |
| qa_medium | 1000 | 10 | 5000 | Integration tests |
| qa_large | 5000 | 20 | 25000 | Performance tests |
| qa_imbalanced | 500 | 5 | 2500 | Class imbalance tests |

### Dataset Format
```
dataset/
├── train/
│   ├── images/
│   └── annotations.json (COCO format)
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

---

## Appendix C: Sign-off Checklist

### Pre-Production Checklist

```
Backend API:
□ All CRUD operations working
□ Validation errors return proper messages
□ Webhook integration tested
□ Rate limiting configured

Worker:
□ All model variants train successfully
□ SOTA features verified
□ Error handling robust
□ Checkpoint saving works
□ Model upload succeeds

Frontend:
□ All wizard steps functional
□ Validation feedback clear
□ Progress tracking works
□ Error messages user-friendly
□ Responsive design verified

Integration:
□ Full training cycle completes
□ Concurrent trainings work
□ Error recovery tested
□ Performance acceptable

Production Readiness:
□ Logs properly structured
□ Metrics collected
□ Alerts configured
□ Documentation updated
```

---

*Son Güncelleme: 2024-01*
*Versiyon: 1.0*
*Hazırlayan: QA Team*
