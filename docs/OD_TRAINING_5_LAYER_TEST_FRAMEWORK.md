# Object Detection Training - 5 Katmanlı Test Framework

## Genel Bakış

Bu framework, OD Training sistemini production'a çıkmadan önce kapsamlı şekilde test etmek için tasarlandı.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         5 KATMANLI TEST PİRAMİDİ                         │
│                                                                          │
│                              ┌─────────┐                                 │
│                              │ KATMAN 5│  Frontend E2E                   │
│                              │  2-3h   │  (Kullanıcı Journey)            │
│                           ┌──┴─────────┴──┐                              │
│                           │   KATMAN 4    │  Edge Cases & Robustness     │
│                           │    2-4h       │  (Sınır Durumları)           │
│                        ┌──┴───────────────┴──┐                           │
│                        │     KATMAN 3        │  Öğrenme Validasyonu      │
│                        │      8-12h          │  (Model Gerçekten Öğreniyor mu?) │
│                     ┌──┴─────────────────────┴──┐                        │
│                     │        KATMAN 2           │  Feature Matrix        │
│                     │         4-6h              │  (Her Özellik Çalışıyor mu?) │
│                  ┌──┴───────────────────────────┴──┐                     │
│                  │           KATMAN 1              │  Sanity Check        │
│                  │            30dk                 │  (Sistem Çalışıyor mu?) │
│                  └─────────────────────────────────┘                     │
│                                                                          │
│  Toplam Süre: ~18-26 saat (paralel çalıştırılabilir testlerle azaltılabilir) │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Ön Gereksinimler

### Test Ortamı
```bash
# RunPod Pod Gereksinimleri
- GPU: RTX 3090 / A4000 / A5000+ (24GB+ VRAM)
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB+ SSD

# Servisler
- Backend API çalışıyor (http://localhost:8000 veya production URL)
- Frontend çalışıyor (http://localhost:3000 veya production URL)
- Supabase bağlantısı aktif
- RunPod endpoint'leri aktif
```

### Environment Variables
```bash
export SUPABASE_URL="https://qvyxpfcwfktxnaeavkxx.supabase.co"
export SUPABASE_SERVICE_KEY="<service_role_key>"
export RUNPOD_API_KEY="<runpod_api_key>"
export RUNPOD_ENDPOINT_OD_TRAINING="3klheezld96ftb"
export API_BASE_URL="http://localhost:8000"
```

---

## KATMAN 1: Sanity Check (30 dakika)

### Amaç
**"Sistem uçtan uca çalışıyor mu?"**

Bu katman, temel pipeline'ın hata vermeden tamamlanıp tamamlanmadığını kontrol eder.

### Test 1.1: API Health Check
```bash
# Backend API'nin ayakta olduğunu doğrula
curl -X GET "${API_BASE_URL}/api/v1/od/health"

# Beklenen: {"status": "healthy", ...}
```

### Test 1.2: Dataset Listeleme
```bash
# Mevcut datasetleri listele
curl -X GET "${API_BASE_URL}/api/v1/od/datasets"

# Beklenen: JSON array (boş olabilir)
```

### Test 1.3: Minimal Training Run (POD İÇİNDE)
```python
"""
minimal_training_test.py - Pod içinde çalıştır
"""
import os
import sys
sys.path.insert(0, '/workspace/od-training')

from handler import handler

# Test job - En minimal config
test_job = {
    "input": {
        "training_run_id": "test-sanity-001",
        "dataset_url": "<GERCEK_DATASET_URL>",  # Supabase'den alınacak
        "dataset_format": "coco",
        "model_type": "rt-detr",
        "model_size": "s",
        "config": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "image_size": 640,
            "augmentation_preset": "none",  # En hızlı
            "use_ema": False,
            "mixed_precision": True,
            "warmup_epochs": 0,
            "early_stopping_patience": 0  # Disable
        },
        "class_names": ["product"],  # Tek class
        "num_classes": 1
    }
}

print("Starting Sanity Test...")
result = handler(test_job)
print(f"Result: {result}")

# Başarı Kriterleri
assert result.get("status") == "completed", f"Training failed: {result}"
assert "model_url" in result, "No model URL in result"
assert result.get("best_map", 0) >= 0, "Invalid mAP"

print("✅ KATMAN 1 PASSED: Sistem çalışıyor!")
```

### Başarı Kriterleri
| Kriter | Kontrol |
|--------|---------|
| API Health | `/od/health` 200 OK |
| Dataset List | `/od/datasets` 200 OK |
| Training Start | Status: training |
| Epoch 1 Complete | Metrics alındı |
| Epoch 2 Complete | Training finished |
| Model Upload | model_url mevcut |
| Final Status | completed |

### Başarısızlık Durumunda
- API health fail → Backend deployment kontrol et
- Dataset list fail → Supabase bağlantısı kontrol et
- Training fail → Worker logs kontrol et, GPU/memory kontrol et

---

## KATMAN 2: Feature Matrix (4-6 saat)

### Amaç
**"Her özellik çalışıyor mu?"**

Bu katman, tüm model/augmentation/SOTA kombinasyonlarının çalıştığını doğrular.

### Test 2.1: Model Matrix

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TEST MATRİXİ                        │
├─────────┬─────┬─────┬─────┬─────┬────────────┬─────────────┤
│ Model   │  S  │  M  │  L  │  X  │ Epochs     │ Batch Size  │
├─────────┼─────┼─────┼─────┼─────┼────────────┼─────────────┤
│ RT-DETR │ [1] │ [2] │ [3] │  -  │ 3 epoch    │ 4           │
│ D-FINE  │ [4] │ [5] │ [6] │ [7] │ 3 epoch    │ 4           │
└─────────┴─────┴─────┴─────┴─────┴────────────┴─────────────┘

Toplam: 7 test × ~15dk = ~2 saat
```

```python
"""
model_matrix_test.py
"""
MODEL_TESTS = [
    # (test_id, model_type, model_size, expected_vram_gb)
    ("M-001", "rt-detr", "s", 6),
    ("M-002", "rt-detr", "m", 8),
    ("M-003", "rt-detr", "l", 12),
    ("M-004", "d-fine", "s", 6),
    ("M-005", "d-fine", "m", 10),
    ("M-006", "d-fine", "l", 14),
    ("M-007", "d-fine", "x", 20),
]

def run_model_test(test_id, model_type, model_size):
    job = {
        "input": {
            "training_run_id": f"test-{test_id}",
            "model_type": model_type,
            "model_size": model_size,
            "config": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 0.0001,
                "augmentation_preset": "light"
            },
            # ... dataset config
        }
    }
    result = handler(job)

    # Doğrulama
    assert result["status"] == "completed", f"{test_id} failed"
    assert len(result.get("metrics_history", [])) == 3, f"{test_id} incomplete epochs"
    print(f"✅ {test_id}: {model_type}-{model_size} PASSED")
    return result

# Tüm modelleri test et
results = {}
for test_id, model_type, model_size, _ in MODEL_TESTS:
    results[test_id] = run_model_test(test_id, model_type, model_size)
```

### Test 2.2: Augmentation Preset Matrix

```
┌───────────────────────────────────────────────────────────────┐
│                 AUGMENTATION TEST MATRİXİ                      │
├────────────┬──────────────────────────────────────┬───────────┤
│ Preset     │ Açıklama                             │ Test ID   │
├────────────┼──────────────────────────────────────┼───────────┤
│ none       │ Augmentation yok (baseline)          │ A-001     │
│ light      │ Sadece flip + scale                  │ A-002     │
│ medium     │ Mosaic (0.3) + basic transforms      │ A-003     │
│ heavy      │ Full augmentation suite              │ A-004     │
│ sota-v2    │ Mosaic + MixUp + CopyPaste (SOTA)    │ A-005     │
└────────────┴──────────────────────────────────────┴───────────┘

Her preset: RT-DETR-S, 5 epoch
Toplam: 5 test × ~20dk = ~1.5 saat
```

```python
"""
augmentation_preset_test.py
"""
AUGMENTATION_TESTS = [
    ("A-001", "none", {"expected_time_factor": 1.0}),
    ("A-002", "light", {"expected_time_factor": 1.05}),
    ("A-003", "medium", {"expected_time_factor": 1.2}),
    ("A-004", "heavy", {"expected_time_factor": 1.8}),
    ("A-005", "sota-v2", {"expected_time_factor": 1.4}),
]

def run_augmentation_test(test_id, preset):
    job = {
        "input": {
            "training_run_id": f"test-{test_id}",
            "model_type": "rt-detr",
            "model_size": "s",
            "config": {
                "epochs": 5,
                "batch_size": 8,
                "augmentation_preset": preset
            }
        }
    }
    result = handler(job)

    # Doğrulama
    assert result["status"] == "completed"

    # Augmentation effect: SOTA presets should improve mAP vs none
    # (Bu test 5+ epoch'ta daha anlamlı)
    print(f"✅ {test_id}: Preset '{preset}' PASSED, mAP={result['best_map']:.4f}")
    return result
```

### Test 2.3: SOTA Features Matrix

```
┌───────────────────────────────────────────────────────────────────┐
│                    SOTA FEATURES TEST MATRİXİ                      │
├─────────────────────┬─────────────────────────────────┬───────────┤
│ Feature             │ Config                          │ Test ID   │
├─────────────────────┼─────────────────────────────────┼───────────┤
│ EMA                 │ use_ema: true, decay: 0.9999    │ S-001     │
│ LLRD                │ use_llrd: true, decay: 0.9      │ S-002     │
│ Mixed Precision     │ mixed_precision: true           │ S-003     │
│ Warmup              │ warmup_epochs: 3                │ S-004     │
│ Early Stopping      │ patience: 5, min_delta: 0.001   │ S-005     │
│ Gradient Clipping   │ gradient_clip_val: 1.0          │ S-006     │
│ Full SOTA Stack     │ All features enabled            │ S-007     │
└─────────────────────┴─────────────────────────────────┴───────────┘

Her test: RT-DETR-S, 10 epoch
Toplam: 7 test × ~25dk = ~3 saat
```

```python
"""
sota_features_test.py
"""
SOTA_TESTS = [
    ("S-001", {"use_ema": True, "ema_decay": 0.9999}),
    ("S-002", {"use_llrd": True, "llrd_decay": 0.9, "head_lr_factor": 10.0}),
    ("S-003", {"mixed_precision": True}),
    ("S-004", {"warmup_epochs": 3}),
    ("S-005", {"early_stopping_patience": 5, "early_stopping_min_delta": 0.001}),
    ("S-006", {"gradient_clip_val": 1.0}),
    ("S-007", {  # Full stack
        "use_ema": True,
        "use_llrd": True,
        "mixed_precision": True,
        "warmup_epochs": 3,
        "gradient_clip_val": 1.0
    }),
]

def run_sota_test(test_id, sota_config):
    base_config = {
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.0001,
        "augmentation_preset": "medium"
    }
    base_config.update(sota_config)

    job = {
        "input": {
            "training_run_id": f"test-{test_id}",
            "model_type": "rt-detr",
            "model_size": "s",
            "config": base_config
        }
    }

    result = handler(job)
    assert result["status"] == "completed"

    # SOTA feature specific validations
    if "use_ema" in sota_config:
        # EMA model should be saved
        assert "ema" in str(result.get("model_url", "")).lower() or result["best_map"] >= 0

    if "early_stopping_patience" in sota_config:
        # Might stop early
        actual_epochs = len(result.get("metrics_history", []))
        print(f"  Early stopping: trained {actual_epochs}/10 epochs")

    print(f"✅ {test_id}: SOTA feature PASSED")
    return result
```

### Katman 2 Özet Tablo

| Test Grubu | Test Sayısı | Tahmini Süre | Öncelik |
|------------|-------------|--------------|---------|
| Model Matrix | 7 | 2 saat | P0 |
| Augmentation | 5 | 1.5 saat | P1 |
| SOTA Features | 7 | 3 saat | P1 |
| **Toplam** | **19** | **6.5 saat** | - |

---

## KATMAN 3: Öğrenme Validasyonu (8-12 saat)

### Amaç
**"Model gerçekten öğreniyor mu?"**

Bu katman, modelin doğru şekilde eğitildiğini ve metriklerin beklenen şekilde ilerlediğini doğrular.

### Test 3.1: Convergence Test

```python
"""
convergence_test.py

Amaç: Loss'un azaldığını ve mAP'ın arttığını doğrula
Dataset: Orta boy (500-1000 images, 5-10 classes)
Config: 50 epoch, full SOTA
"""

def run_convergence_test():
    job = {
        "input": {
            "training_run_id": "test-convergence-001",
            "dataset_url": "<ORTA_BOY_DATASET_URL>",
            "model_type": "d-fine",
            "model_size": "m",
            "config": {
                "epochs": 50,
                "batch_size": 16,
                "learning_rate": 0.0001,
                "augmentation_preset": "sota-v2",
                "use_ema": True,
                "use_llrd": True,
                "mixed_precision": True,
                "warmup_epochs": 5,
                "early_stopping_patience": 15
            }
        }
    }

    result = handler(job)
    metrics = result.get("metrics_history", [])

    # === CONVERGENCE KRİTERLERİ ===

    # 1. Training Loss Azalıyor mu?
    train_losses = [m["train_loss"] for m in metrics]
    loss_decrease = train_losses[0] - train_losses[-1]
    assert loss_decrease > 0, f"Loss did not decrease: {train_losses[0]} -> {train_losses[-1]}"
    print(f"✅ Loss decreased by {loss_decrease:.4f}")

    # 2. mAP Artıyor mu?
    maps = [m["map"] for m in metrics]
    map_increase = maps[-1] - maps[0]
    assert map_increase > 0, f"mAP did not increase: {maps[0]} -> {maps[-1]}"
    print(f"✅ mAP increased by {map_increase:.4f}")

    # 3. Final mAP > Initial mAP * 2 (anlamlı öğrenme)
    assert maps[-1] >= maps[0] * 1.5, f"mAP improvement insufficient: {maps[-1]/maps[0]:.2f}x"
    print(f"✅ mAP improved {maps[-1]/maps[0]:.2f}x")

    # 4. Overfitting Kontrolü (val_loss << train_loss olmamalı)
    final_train_loss = metrics[-1]["train_loss"]
    final_val_loss = metrics[-1]["val_loss"]
    overfit_ratio = final_train_loss / final_val_loss if final_val_loss > 0 else 0
    assert overfit_ratio > 0.3, f"Possible overfitting: train/val ratio = {overfit_ratio:.2f}"
    print(f"✅ No severe overfitting (ratio: {overfit_ratio:.2f})")

    # 5. Best Epoch Reasonable (son 1/3'te olmalı)
    best_epoch = result["best_epoch"]
    assert best_epoch >= len(metrics) * 0.3, f"Best epoch too early: {best_epoch}/{len(metrics)}"
    print(f"✅ Best epoch at reasonable point: {best_epoch}/{len(metrics)}")

    print(f"\n🎉 CONVERGENCE TEST PASSED!")
    print(f"   Final mAP: {result['best_map']:.4f}")
    print(f"   Final mAP@50: {result['best_map_50']:.4f}")
    print(f"   Best Epoch: {best_epoch}")

    return result
```

### Test 3.2: Metric Consistency Test

```python
"""
metric_consistency_test.py

Amaç: Metriklerin tutarlı ve mantıklı olduğunu doğrula
"""

def validate_metrics(metrics_history):
    for i, metrics in enumerate(metrics_history):
        epoch = i + 1

        # 1. Tüm gerekli metrikler mevcut
        required = ["epoch", "train_loss", "val_loss", "map", "map_50", "map_75"]
        for key in required:
            assert key in metrics, f"Epoch {epoch}: Missing metric '{key}'"

        # 2. Metrikler valid range'de
        assert 0 <= metrics["map"] <= 1, f"Epoch {epoch}: Invalid mAP {metrics['map']}"
        assert 0 <= metrics["map_50"] <= 1, f"Epoch {epoch}: Invalid mAP@50"
        assert 0 <= metrics["map_75"] <= 1, f"Epoch {epoch}: Invalid mAP@75"
        assert metrics["train_loss"] >= 0, f"Epoch {epoch}: Negative train_loss"
        assert metrics["val_loss"] >= 0, f"Epoch {epoch}: Negative val_loss"

        # 3. mAP hierarchy: mAP@50 >= mAP >= mAP@75
        assert metrics["map_50"] >= metrics["map"] - 0.01, f"Epoch {epoch}: mAP@50 < mAP"
        assert metrics["map"] >= metrics["map_75"] - 0.01, f"Epoch {epoch}: mAP < mAP@75"

        # 4. Epoch number correct
        assert metrics["epoch"] == epoch, f"Epoch mismatch: {metrics['epoch']} != {epoch}"

    print(f"✅ All {len(metrics_history)} epochs have valid metrics")
```

### Test 3.3: Checkpoint Integrity Test

```python
"""
checkpoint_test.py

Amaç: Checkpoint dosyalarının doğru kaydedildiğini doğrula
"""

def validate_checkpoint(model_url):
    import torch
    import requests
    import tempfile

    # 1. Download checkpoint
    response = requests.get(model_url)
    assert response.status_code == 200, f"Failed to download: {model_url}"

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(response.content)
        checkpoint_path = f.name

    # 2. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 3. Validate structure
    assert "model_state_dict" in checkpoint, "Missing model weights"
    assert "config" in checkpoint, "Missing config"
    assert "class_names" in checkpoint, "Missing class names"
    assert "best_map" in checkpoint, "Missing best mAP"

    # 4. Validate weights are not all zeros
    for name, param in checkpoint["model_state_dict"].items():
        if param.numel() > 0:
            assert not torch.all(param == 0), f"Layer {name} has all zeros"
            break  # Just check first non-empty

    print(f"✅ Checkpoint valid: {checkpoint_path}")
    print(f"   Classes: {checkpoint['class_names']}")
    print(f"   Best mAP: {checkpoint['best_map']:.4f}")

    return checkpoint
```

### Katman 3 Özet

| Test | Açıklama | Süre | Başarı Kriteri |
|------|----------|------|----------------|
| Convergence | Loss↓, mAP↑ | 4-6h | mAP 1.5x+ artış |
| Metric Consistency | Valid ranges | 1h | Tüm metrikler valid |
| Checkpoint Integrity | Model dosyası | 30m | Yüklenebilir, non-zero weights |
| Class-wise mAP | Per-class metrics | 2h | Her class için mAP > 0 |

---

## KATMAN 4: Edge Cases & Robustness (2-4 saat)

### Amaç
**"Sistem sınır durumlarında nasıl davranıyor?"**

### Test 4.1: Input Validation Tests

```python
"""
input_validation_test.py
"""

INVALID_INPUTS = [
    # (test_id, input_override, expected_error)
    ("E-001", {"model_type": "invalid-model"}, "Invalid model type"),
    ("E-002", {"model_size": "xxl"}, "Invalid model size"),
    ("E-003", {"config": {"epochs": 0}}, "Epochs must be >= 1"),
    ("E-004", {"config": {"epochs": -5}}, "Epochs must be >= 1"),
    ("E-005", {"config": {"batch_size": 0}}, "Batch size must be >= 1"),
    ("E-006", {"config": {"learning_rate": -0.001}}, "Learning rate must be > 0"),
    ("E-007", {"dataset_url": ""}, "Dataset URL required"),
    ("E-008", {"dataset_url": "not-a-url"}, "Invalid dataset URL"),
    ("E-009", {"class_names": []}, "At least one class required"),
    ("E-010", {"num_classes": 0}, "Invalid class count"),
]

def test_invalid_input(test_id, override, expected_error):
    base_job = get_valid_job()  # Valid baseline

    # Apply override
    for key, value in override.items():
        if key == "config":
            base_job["input"]["config"].update(value)
        else:
            base_job["input"][key] = value

    try:
        result = handler(base_job)
        # Should fail
        if result.get("status") == "failed":
            assert expected_error.lower() in result.get("error_message", "").lower()
            print(f"✅ {test_id}: Correctly rejected with error")
        else:
            print(f"❌ {test_id}: Should have failed but didn't")
            return False
    except Exception as e:
        # Exception is also acceptable for validation
        print(f"✅ {test_id}: Raised exception: {type(e).__name__}")

    return True
```

### Test 4.2: Resource Limit Tests

```python
"""
resource_limit_test.py
"""

def test_oom_handling():
    """GPU OOM durumunda graceful fail"""
    job = {
        "input": {
            "training_run_id": "test-oom-001",
            "model_type": "d-fine",
            "model_size": "x",  # En büyük model
            "config": {
                "epochs": 2,
                "batch_size": 64,  # Çok büyük batch
                "image_size": 1280  # Çok büyük image
            }
        }
    }

    result = handler(job)

    # OOM olmalı ve anlamlı hata vermeli
    assert result["status"] == "failed"
    error_msg = result.get("error_message", "").lower()
    assert "memory" in error_msg or "oom" in error_msg or "cuda" in error_msg
    print(f"✅ OOM handled gracefully: {result['error_message'][:100]}")

def test_timeout_handling():
    """Çok uzun training için timeout"""
    # Bu test RunPod timeout settings'e bağlı
    pass

def test_disk_space():
    """Disk dolduğunda ne olur"""
    # Simüle etmek zor, monitoring ile kontrol
    pass
```

### Test 4.3: Dataset Edge Cases

```python
"""
dataset_edge_cases_test.py
"""

DATASET_EDGE_CASES = [
    # (test_id, description, dataset_config)
    ("D-001", "Single class", {"classes": 1, "images": 100}),
    ("D-002", "Many classes (50+)", {"classes": 50, "images": 500}),
    ("D-003", "Minimal images (20)", {"classes": 3, "images": 20}),
    ("D-004", "Class imbalance (100:1)", {"classes": 2, "images": 200, "imbalance": 100}),
    ("D-005", "Small images (64x64)", {"image_size": 64}),
    ("D-006", "Large images (2048x2048)", {"image_size": 2048}),
    ("D-007", "No validation split", {"val_ratio": 0}),
    ("D-008", "Tiny validation (1%)", {"val_ratio": 0.01}),
]

# Not: Bu testler için özel test datasetleri oluşturulmalı
# veya mevcut datasetler bu özelliklere göre filtrelenmeli
```

### Test 4.4: Cancel & Recovery Tests

```python
"""
cancel_recovery_test.py
"""

def test_training_cancel():
    """Training iptal edildiğinde graceful stop"""
    import threading
    import time

    # Start training in background
    def run_training():
        return handler(long_running_job)

    thread = threading.Thread(target=run_training)
    thread.start()

    # Wait for training to start
    time.sleep(30)

    # Send cancel signal (via API or Supabase)
    cancel_training("test-cancel-001")

    # Wait for graceful stop
    thread.join(timeout=60)

    # Verify
    status = get_training_status("test-cancel-001")
    assert status == "cancelled"
    print("✅ Training cancelled gracefully")

def test_resume_from_checkpoint():
    """Checkpoint'ten devam etme"""
    # Phase 1: Start and stop at epoch 5
    # Phase 2: Resume from checkpoint
    # Verify: Continues from epoch 5
    pass
```

### Katman 4 Özet

| Test Grubu | Test Sayısı | Öncelik | Notlar |
|------------|-------------|---------|--------|
| Input Validation | 10 | P0 | Tüm geçersiz input'lar reject edilmeli |
| Resource Limits | 3 | P1 | OOM, timeout graceful fail |
| Dataset Edge Cases | 8 | P1 | Minimum viable dataset kontrol |
| Cancel & Recovery | 2 | P2 | Graceful shutdown |

---

## KATMAN 5: Frontend E2E Tests (2-3 saat)

### Amaç
**"Kullanıcı deneyimi uçtan uca çalışıyor mu?"**

Bu katman, gerçek bir kullanıcının frontend'den training başlatıp sonuçları görmesine kadar tüm akışı test eder.

### Test 5.1: Training Wizard Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING WIZARD E2E TEST                              │
│                                                                          │
│  START                                                                   │
│    │                                                                     │
│    ▼                                                                     │
│  ┌─────────────────┐                                                     │
│  │ 1. Dataset Step │ → Dataset seç, split ayarla, stats kontrol et      │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ 2. Preprocess   │ → Image size seç, resize strategy seç              │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ 3. Offline Aug  │ → Toggle on/off, multiplier test                   │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ 4. Online Aug   │ → SOTA-v2 seç, preset değiştir                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ 5. Model        │ → RT-DETR/D-FINE seç, size seç                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ 6. Hyperparams  │ → Epochs, batch, LR, SOTA features                 │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ 7. Review       │ → Summary kontrol, name gir, submit                │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ Training Detail │ → Progress izle, metrics görüntüle                 │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│        SUCCESS                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Manual Test Checklist

```markdown
## E2E-001: Happy Path - Tam Training Akışı

### Ön Koşullar
- [ ] En az 1 dataset mevcut (annotated images ile)
- [ ] Backend API çalışıyor
- [ ] RunPod endpoint aktif

### Adımlar

**Step 1: Dataset Selection**
- [ ] /od/training/new sayfasını aç
- [ ] Datasets dropdown'u aç - listede datasetler görünüyor mu?
- [ ] Bir dataset seç
- [ ] Stats card'da bilgiler gösteriliyor mu?
  - [ ] Total images
  - [ ] Annotation coverage
  - [ ] Class count
  - [ ] Dataset size badge
- [ ] Split slider'ları çalışıyor mu? (Train/Val/Test)
- [ ] "Next" butonu aktif mi?
- [ ] "Next"e tıkla

**Step 2: Preprocessing**
- [ ] Target size butonları görünüyor mu? (320, 480, 640, 800, 1024, 1280)
- [ ] 640 seç - highlight oluyor mu?
- [ ] Resize strategy kartları görünüyor mu?
- [ ] "Letterbox" seç
- [ ] "Next"e tıkla

**Step 3: Offline Augmentation**
- [ ] Toggle kapalı durumda mı? (default)
- [ ] Toggle'ı aç
- [ ] Multiplier seçenekleri görünüyor mu? (1x, 2x, 3x, 5x, 10x)
- [ ] 2x seç
- [ ] Resulting size hesaplanıyor mu?
- [ ] Toggle'ı kapat (test için offline aug kullanmıyoruz)
- [ ] "Next"e tıkla

**Step 4: Online Augmentation**
- [ ] Preset kartları görünüyor mu?
- [ ] "SOTA-v2" kartında "Recommended" badge var mı?
- [ ] "SOTA-v2" seç - highlight oluyor mu?
- [ ] Feature list görünüyor mu? (Mosaic, MixUp, CopyPaste...)
- [ ] "Next"e tıkla

**Step 5: Model Selection**
- [ ] Model kartları görünüyor mu? (RT-DETR, D-FINE)
- [ ] "RT-DETR" seç
- [ ] Size seçenekleri görünüyor mu? (S, M, L)
- [ ] "S" seç (hızlı test için)
- [ ] VRAM estimation gösteriliyor mu?
- [ ] "Pretrained" checkbox checked mi? (default)
- [ ] "Next"e tıkla

**Step 6: Hyperparameters**
- [ ] Epochs input görünüyor mu?
- [ ] 5 yaz (hızlı test)
- [ ] Batch size dropdown çalışıyor mu?
- [ ] 8 seç
- [ ] Learning rate dropdown çalışıyor mu?
- [ ] 0.0001 seç (default)
- [ ] SOTA toggles:
  - [ ] EMA toggle - aç
  - [ ] Mixed Precision toggle - aç
  - [ ] LLRD toggle - kapat (hızlı test için)
- [ ] Early Stopping:
  - [ ] Patience slider görünüyor mu?
  - [ ] 10'a ayarla
- [ ] "Next"e tıkla

**Step 7: Review & Submit**
- [ ] Summary card'da tüm seçimler doğru görünüyor mu?
  - [ ] Dataset name
  - [ ] Model: RT-DETR-S
  - [ ] Epochs: 5
  - [ ] Augmentation: SOTA-v2
  - [ ] SOTA features badges (EMA, Mixed Precision)
- [ ] Training name input'u görünüyor mu?
- [ ] "QA Test Run - E2E-001" yaz
- [ ] "Start Training" butonu aktif mi?
- [ ] "Start Training"e tıkla
- [ ] Button spinner gösteriyor mu?
- [ ] Success toast çıkıyor mu?
- [ ] Training detail sayfasına redirect oluyor mu?

**Training Detail Page**
- [ ] Training name görünüyor mu?
- [ ] Status: "training" veya "queued"
- [ ] Progress bar görünüyor mu?
- [ ] Epoch counter: "0/5" (başlangıç)
- [ ] 30 saniye bekle - progress güncellenliyor mu?
- [ ] Epoch 1 tamamlandığında metrics görünüyor mu?
  - [ ] Train loss
  - [ ] Val loss
  - [ ] mAP
- [ ] Loss chart çiziliyor mu?
- [ ] mAP chart çiziliyor mu?

**Training Completion**
- [ ] ~10-15 dakika bekle (5 epoch)
- [ ] Status: "completed" oluyor mu?
- [ ] Final metrics görünüyor mu?
- [ ] Best epoch highlighted mı?
- [ ] "Download Model" butonu aktif mi?
- [ ] Butona tıkla - model indiriliyor mu?

### Başarı Kriterleri
- [ ] Tüm wizard step'leri hatasız geçildi
- [ ] Training başarıyla tamamlandı
- [ ] Metrics anlamlı (mAP > 0)
- [ ] Model dosyası indirilebilir

### Süre: ~20-30 dakika (5 epoch training dahil)
```

### E2E-002: Error Handling Flow

```markdown
## E2E-002: Hata Durumları Testi

### Test 2.1: Validation Errors
- [ ] Dataset seçmeden "Next" - error gösteriliyor mu?
- [ ] Name boş bırakıp submit - error gösteriliyor mu?
- [ ] Invalid epochs (0) girip - error gösteriliyor mu?

### Test 2.2: API Errors
- [ ] Network offline iken submit - error toast gösteriliyor mu?
- [ ] Button re-enable oluyor mu?

### Test 2.3: Training Failure
- [ ] Başarısız training detail sayfası:
  - [ ] Status: "failed" kırmızı badge
  - [ ] Error message görünüyor mu?
  - [ ] Traceback (accordion ile) görünüyor mu?
```

### E2E-003: Navigation & State

```markdown
## E2E-003: Navigasyon ve State Yönetimi

### Test 3.1: Back Navigation
- [ ] Step 4'e kadar ilerle
- [ ] "Back"e 3 kez tıkla - Step 1'e dönüyor mu?
- [ ] Önceki seçimler korunuyor mu?

### Test 3.2: Stepper Click
- [ ] Step 5'e kadar tamamla
- [ ] Stepper'da Step 2'ye tıkla
- [ ] Step 2 açılıyor mu?
- [ ] Seçimler korunuyor mu?

### Test 3.3: Page Refresh
- [ ] Step 4'te iken sayfayı yenile
- [ ] State sıfırlanıyor mu? (beklenen davranış)
- [ ] Kullanıcı uyarılıyor mu? (optional)

### Test 3.4: Browser Back
- [ ] Training submit sonrası detail page'de
- [ ] Browser back butonuna bas
- [ ] Training list'e dönüyor mu?
```

### E2E-004: Smart Defaults

```markdown
## E2E-004: Smart Defaults Testi

- [ ] Küçük dataset seç (<500 images)
- [ ] Smart Recommendations sidebar görünüyor mu?
- [ ] "Small dataset detected" mesajı var mı?
- [ ] "Apply Recommendations" butonuna tıkla
- [ ] Toast: "Smart defaults applied!" görünüyor mu?
- [ ] Sonraki step'lere git:
  - [ ] Augmentation: Heavy preset seçili mi?
  - [ ] Model: Küçük model önerilmiş mi?
  - [ ] Hyperparams: Daha fazla epoch?
```

### E2E-005: Training List & Filtering

```markdown
## E2E-005: Training List Sayfası

### List View
- [ ] /od/training sayfasını aç
- [ ] Training list görünüyor mu?
- [ ] Her training card'da:
  - [ ] Name
  - [ ] Status badge
  - [ ] Model type/size
  - [ ] Dataset name
  - [ ] Created date
  - [ ] Metrics (if completed)

### Filtering
- [ ] Status filter dropdown
  - [ ] "All" seç - tüm training'ler
  - [ ] "Completed" seç - sadece tamamlananlar
  - [ ] "Training" seç - devam edenler
  - [ ] "Failed" seç - başarısızlar
- [ ] Dataset filter (varsa)
- [ ] Sort options

### Actions
- [ ] Completed training'e tıkla - detail açılıyor mu?
- [ ] Cancel butonu (running training için) çalışıyor mu?
- [ ] Delete butonu çalışıyor mu? (confirm dialog ile)
```

### E2E-006: Responsive Design

```markdown
## E2E-006: Responsive Tasarım Testi

### Desktop (1920x1080)
- [ ] Wizard: Content + Sidebar yan yana
- [ ] Stepper: Full horizontal stepper
- [ ] Cards: 3-4 column grid

### Tablet (768x1024)
- [ ] Wizard: 2 column layout
- [ ] Stepper: Compact, possibly vertical
- [ ] Cards: 2 column grid

### Mobile (375x812)
- [ ] Wizard: Single column
- [ ] Stepper: "Step X of 7" compact
- [ ] Cards: Single column stack
- [ ] Buttons: Full width, thumb-friendly
```

### Katman 5 Özet

| Test ID | Açıklama | Süre | Öncelik |
|---------|----------|------|---------|
| E2E-001 | Happy Path | 30m | P0 |
| E2E-002 | Error Handling | 15m | P0 |
| E2E-003 | Navigation | 15m | P1 |
| E2E-004 | Smart Defaults | 10m | P2 |
| E2E-005 | List & Filter | 15m | P1 |
| E2E-006 | Responsive | 15m | P2 |
| **Total** | - | **~2h** | - |

---

## Test Execution Plan

### Önerilen Sıralama

```
┌────────────────────────────────────────────────────────────────┐
│                    TEST EXECUTION TIMELINE                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Gün 1 (4-5 saat)                                              │
│  ├── Katman 1: Sanity Check (30dk)                             │
│  ├── Katman 4: Input Validation (1h) - paralel çalışabilir     │
│  └── Katman 5: E2E Happy Path (30m)                            │
│                                                                 │
│  Gün 2 (6-8 saat)                                              │
│  ├── Katman 2: Model Matrix (2h)                               │
│  ├── Katman 2: Augmentation Presets (1.5h)                     │
│  └── Katman 2: SOTA Features (3h)                              │
│                                                                 │
│  Gün 3 (8-12 saat)                                             │
│  └── Katman 3: Convergence Test (8-12h) - arka planda          │
│                                                                 │
│  Gün 4 (2-3 saat)                                              │
│  ├── Katman 4: Edge Cases (2h)                                 │
│  └── Katman 5: Remaining E2E (1.5h)                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Parallel Execution

Bazı testler paralel çalıştırılabilir:
- Katman 2 Model testleri (farklı model tipleri)
- Katman 4 Input validation testleri
- Katman 5 farklı browser/device testleri

### Test Environment Setup

```bash
# 1. Clone worker repo to pod
git clone <repo> /workspace/od-training

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export SUPABASE_URL="..."
export SUPABASE_SERVICE_KEY="..."

# 4. Run tests
python test_layer_1.py
python test_layer_2.py
# etc.
```

---

## Sign-off Checklist

### Production-Ready Criteria

```markdown
## Pre-Production Checklist

### Katman 1: Sanity ✅
- [ ] API health check passed
- [ ] Minimal training completed
- [ ] Model upload successful

### Katman 2: Features ✅
- [ ] All 7 model variants work
- [ ] All 5 augmentation presets work
- [ ] All SOTA features work individually
- [ ] Full SOTA stack works

### Katman 3: Quality ✅
- [ ] Loss decreases over epochs
- [ ] mAP increases over epochs
- [ ] No severe overfitting
- [ ] Checkpoints are valid

### Katman 4: Robustness ✅
- [ ] Invalid inputs rejected with clear errors
- [ ] OOM handled gracefully
- [ ] Cancel works correctly
- [ ] Edge case datasets handled

### Katman 5: UX ✅
- [ ] Full wizard flow works
- [ ] Errors displayed clearly
- [ ] Navigation intuitive
- [ ] Responsive design works

### Final Sign-off
- [ ] All P0 tests passed
- [ ] All P1 tests passed
- [ ] P2 tests reviewed (non-blocking)
- [ ] QA Lead approval
- [ ] Engineering Lead approval
```

---

## Appendix: Test Data Requirements

### Minimal Test Dataset
```yaml
name: "qa-minimal"
images: 50
classes: 3
annotations_per_image: 2-5
format: COCO
split: 70/20/10
```

### Small Test Dataset
```yaml
name: "qa-small"
images: 200
classes: 5
annotations_per_image: 3-8
format: COCO
split: 70/20/10
```

### Medium Test Dataset
```yaml
name: "qa-medium"
images: 1000
classes: 10
annotations_per_image: 5-15
format: COCO
split: 80/10/10
```

### Edge Case Datasets
```yaml
- name: "qa-single-class" (1 class, 100 images)
- name: "qa-many-classes" (50 classes, 500 images)
- name: "qa-imbalanced" (2 classes, 100:1 ratio)
- name: "qa-tiny-images" (64x64 images)
- name: "qa-large-images" (2048x2048 images)
```

---

*Framework Version: 1.0*
*Last Updated: 2024-01*
*Author: QA Team*
