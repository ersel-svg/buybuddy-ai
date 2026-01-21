# SOTA Object Detection Fine-Tuning Implementation Plan

## Overview

Bu doküman, OD Training Worker'ına SOTA (State-of-the-Art) fine-tuning özelliklerinin eklenmesi için detaylı implementasyon planını içerir.

**Hedef:** Mevcut sisteme entegre, production-ready, SOTA seviye OD model fine-tuning pipeline'ı.

**Desteklenen Modeller:**
- RT-DETR (Real-Time DETR) - Apache 2.0
- D-FINE (USTC) - Apache 2.0
- RF-DETR (Roboflow) - Apache 2.0

**Kaldırılan Modeller:**
- YOLO-NAS (Lisans sorunu - AGPL)

---

## Mevcut Sistem Entegrasyonu

### Değişmeyecek Bileşenler
| Bileşen | Dosya | Neden |
|---------|-------|-------|
| Training API | `apps/api/src/api/v1/od/training.py` | Endpoint'ler çalışıyor |
| Models API | `apps/api/src/api/v1/od/models.py` | Model registry çalışıyor |
| Export Service | `apps/api/src/services/od_export.py` | COCO/YOLO export çalışıyor |
| Database | `od_training_runs`, `od_trained_models` | Schema yeterli |
| Storage | `od-models`, `od-training-data` buckets | Mevcut |
| RunPod Handler | `workers/od-training/handler.py` | Minimal değişiklik |

### Güncellenecek Bileşenler
| Bileşen | Dosya | Değişiklik |
|---------|-------|------------|
| Config | `config.py` | SOTA config options ekleme |
| Base Trainer | `trainers/base.py` | SOTA features entegrasyonu |
| RT-DETR Trainer | `trainers/rt_detr.py` | Yeniden yazılacak |
| Handler | `handler.py` | D-FINE dispatch ekleme |

### Yeni Eklenecek Bileşenler
| Bileşen | Konum | İçerik |
|---------|-------|--------|
| Augmentations | `src/augmentations/` | Mosaic, MixUp, CopyPaste |
| Losses | `src/losses/` | Focal, CIoU, DFL |
| Training Utils | `src/training/` | EMA, LLRD, Scheduler |
| Evaluation | `src/evaluation/` | COCO mAP |
| Dataset | `src/dataset.py` | Enhanced COCO loader |
| D-FINE Trainer | `trainers/dfine.py` | Yeni trainer |

---

## Dosya Yapısı

```
workers/od-training/
├── handler.py                      # Minimal güncelleme
├── config.py                       # SOTA config ekleme
├── requirements.txt                # Yeni dependencies
├── Dockerfile                      # Güncelleme
│
├── trainers/
│   ├── __init__.py
│   ├── base.py                     # SOTA base trainer
│   ├── rt_detr.py                  # Yeniden yazılacak
│   ├── dfine.py                    # Yeni
│   └── rf_detr.py                  # Güncelleme
│
└── src/
    ├── __init__.py
    │
    ├── augmentations/
    │   ├── __init__.py
    │   ├── mosaic.py               # Mosaic augmentation
    │   ├── mixup.py                # MixUp augmentation
    │   ├── copypaste.py            # Copy-Paste augmentation
    │   └── transforms.py           # Albumentations pipeline
    │
    ├── losses/
    │   ├── __init__.py
    │   ├── focal.py                # Focal Loss
    │   ├── iou.py                  # GIoU, DIoU, CIoU, SIoU
    │   └── dfl.py                  # Distribution Focal Loss
    │
    ├── training/
    │   ├── __init__.py
    │   ├── ema.py                  # Exponential Moving Average
    │   ├── optimizer.py            # LLRD optimizer builder
    │   └── scheduler.py            # Warmup + Cosine scheduler
    │
    ├── evaluation/
    │   ├── __init__.py
    │   └── coco_eval.py            # COCO mAP evaluation
    │
    └── dataset.py                  # Enhanced COCO dataset
```

---

## Task 1: Training Utilities (EMA, LLRD, Scheduler)

### Task 1.1: EMA (Exponential Moving Average)

**Dosya:** `src/training/ema.py`

**Açıklama:**
Model ağırlıklarının hareketli ortalamasını tutar. Training sırasında weights salınır, EMA smooth bir versiyon tutar. Inference/validation'da EMA weights kullanılır.

**Formül:**
```
θ_ema = decay × θ_ema + (1 - decay) × θ_current
decay = 0.9999 (tipik)
```

**Özellikler:**
- Warmup decay: İlk step'lerde düşük decay
- Shadow weights: Ayrı EMA weights
- Apply/restore: Validation için weights swap
- State dict save/load: Checkpoint uyumluluğu

**Acceptance Criteria:**
- [ ] `ModelEMA` class'ı oluşturuldu
- [ ] `update(model)` methodu her step'te çağrılabilir
- [ ] `apply_shadow(model)` weights'i EMA ile değiştirir
- [ ] `restore(model)` orijinal weights'e döner
- [ ] Warmup decay ilk 2000 step'te uygulanır
- [ ] State dict checkpoint'a kaydedilebilir/yüklenebilir
- [ ] Unit test: EMA decay doğru hesaplanıyor

**Beklenen Etki:** +0.5-2% mAP

---

### Task 1.2: LLRD (Layer-wise Learning Rate Decay)

**Dosya:** `src/training/optimizer.py`

**Açıklama:**
Derin katmanlara düşük, sığ katmanlara yüksek learning rate uygular. Pretrained backbone'u korur, yeni head'i agresif eğitir.

**Formül:**
```
lr(layer_i) = base_lr × decay^(num_layers - i - 1)
decay = 0.9 (tipik)
```

**Özellikler:**
- Layer gruplarına göre LR ayarı
- Backbone stages: Kademeli decay
- Encoder/Decoder: Base LR
- Detection Head: 10x base LR
- Weight decay grupları

**Acceptance Criteria:**
- [ ] `build_llrd_optimizer(model, base_lr, decay)` fonksiyonu
- [ ] RT-DETR için layer grupları tanımlı
- [ ] D-FINE için layer grupları tanımlı
- [ ] Backbone stages farklı LR alıyor
- [ ] Head katmanları 10x LR alıyor
- [ ] AdamW optimizer döndürüyor
- [ ] Unit test: Layer LR'ları doğru hesaplanıyor

**Beklenen Etki:** +0.5-1% mAP

---

### Task 1.3: LR Scheduler (Warmup + Cosine)

**Dosya:** `src/training/scheduler.py`

**Açıklama:**
İlk epoch'larda warmup (lineer artış), sonra cosine annealing (smooth azalma).

**Formül:**
```
Warmup:  lr = base_lr × (step / warmup_steps)
Cosine:  lr = min_lr + 0.5 × (base_lr - min_lr) × (1 + cos(π × progress))
```

**Özellikler:**
- Linear warmup (ilk 3 epoch veya 1000 step)
- Cosine annealing (geri kalan)
- Per-step update (epoch değil)
- Min LR: base_lr × 0.01

**Acceptance Criteria:**
- [ ] `build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch)` fonksiyonu
- [ ] Warmup period'da LR linear artıyor
- [ ] Warmup sonrası cosine decay başlıyor
- [ ] Per-step update destekleniyor
- [ ] `get_last_lr()` methodu çalışıyor
- [ ] Unit test: LR curve doğru şekilde oluşuyor

**Beklenen Etki:** Stabil training, daha iyi convergence

---

## Task 2: Loss Functions

### Task 2.1: Focal Loss

**Dosya:** `src/losses/focal.py`

**Açıklama:**
Class imbalance problemi için. Easy samples'ı down-weight eder, hard samples'a odaklanır.

**Formül:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
γ = 2.0 (focusing parameter)
α = 0.25 (class balance)
```

**Özellikler:**
- Configurable gamma ve alpha
- Multi-class support
- Label smoothing opsiyonel
- Reduction: mean/sum/none

**Acceptance Criteria:**
- [ ] `FocalLoss(gamma=2.0, alpha=0.25)` class'ı
- [ ] Forward: (pred_logits, targets) → scalar loss
- [ ] Easy samples (high confidence) düşük loss
- [ ] Hard samples (low confidence) yüksek loss
- [ ] Label smoothing opsiyonu
- [ ] Unit test: Focal vs CE karşılaştırması

**Beklenen Etki:** Daha iyi class balance, +1-2% mAP (imbalanced data için)

---

### Task 2.2: IoU Losses (GIoU, DIoU, CIoU)

**Dosya:** `src/losses/iou.py`

**Açıklama:**
Bbox regression için IoU-based loss'lar. CIoU en iyi performans.

**Formüller:**
```
IoU  = Intersection / Union
GIoU = IoU - (C - U) / C
DIoU = IoU - d²/c²
CIoU = DIoU - αv  (v = aspect ratio difference)
```

**Özellikler:**
- GIoU, DIoU, CIoU, SIoU variants
- Batch processing
- Box format: xyxy veya cxcywh
- Gradient-safe (eps handling)

**Acceptance Criteria:**
- [ ] `compute_iou(boxes1, boxes2, iou_type)` fonksiyonu
- [ ] `CIoULoss()` class'ı
- [ ] XYXY ve CXCYWH format desteği
- [ ] Batch processing (NxM IoU matrix)
- [ ] Gradient flow doğru (no in-place ops)
- [ ] Unit test: Bilinen box'lar için IoU doğru

**Beklenen Etki:** Daha iyi bbox regression, +0.5-1% mAP

---

### Task 2.3: Distribution Focal Loss (DFL)

**Dosya:** `src/losses/dfl.py`

**Açıklama:**
Bbox kenarlarını distribution olarak predict eder. RT-DETR ve D-FINE için önemli.

**Formül:**
```
E[x] = Σ(i × softmax(logits[i]))  # Expected value
DFL = -(y_{i+1} - y) × log(p_i) - (y - y_i) × log(p_{i+1})
```

**Özellikler:**
- Configurable bin count (default: 16)
- Integral hesaplama
- Smooth regression

**Acceptance Criteria:**
- [ ] `DFLoss(num_bins=16)` class'ı
- [ ] Distribution → expected value dönüşümü
- [ ] Ground truth bin interpolation
- [ ] Unit test: Bilinen distribution için loss doğru

**Beklenen Etki:** Daha hassas bbox, +0.5% mAP

---

## Task 3: Augmentations

### Task 3.1: Mosaic Augmentation

**Dosya:** `src/augmentations/mosaic.py`

**Açıklama:**
4 resmi 2x2 grid'de birleştirir. Her resmin bbox'ları yeni pozisyona göre transform edilir.

**Özellikler:**
- 4 random image selection
- Random center point
- BBox coordinate transformation
- Out-of-bounds bbox clipping
- Min bbox size filtering

**Acceptance Criteria:**
- [ ] `MosaicAugmentation(img_size=640)` class'ı
- [ ] `__call__(images, targets) → image, target`
- [ ] 4 resim doğru birleştiriliyor
- [ ] BBox koordinatları doğru transform ediliyor
- [ ] Görüntü dışına çıkan bbox'lar clip ediliyor
- [ ] Çok küçük bbox'lar filtreleniyor
- [ ] Visual test: Örnek mosaic output

**Beklenen Etki:** +2-5% mAP

---

### Task 3.2: MixUp Augmentation

**Dosya:** `src/augmentations/mixup.py`

**Açıklama:**
2 resmi alpha oranında blend eder. Her iki resmin bbox'ları sonuca dahil edilir.

**Formül:**
```
image = α × image_a + (1-α) × image_b
α ~ Beta(8.0, 8.0)
```

**Özellikler:**
- Beta distribution sampling
- Image blending
- Target merging (both images' boxes)
- Configurable alpha range

**Acceptance Criteria:**
- [ ] `MixUpAugmentation(alpha=8.0)` class'ı
- [ ] Alpha değeri Beta distribution'dan sample
- [ ] Pixel values doğru blend ediliyor
- [ ] Her iki image'ın bbox'ları merge ediliyor
- [ ] Visual test: Örnek mixup output

**Beklenen Etki:** +0.5-1% mAP, regularization

---

### Task 3.3: Copy-Paste Augmentation

**Dosya:** `src/augmentations/copypaste.py`

**Açıklama:**
Bir resimden objeleri kesip başka resme yapıştırır. Nadir sınıfları artırmak için ideal.

**Özellikler:**
- Instance segmentation mask kullanımı (varsa)
- Bbox-based fallback
- Random position placement
- Occlusion handling
- Scale jittering

**Acceptance Criteria:**
- [ ] `CopyPasteAugmentation(prob=0.5)` class'ı
- [ ] Source image'dan obje extraction
- [ ] Target image'a placement
- [ ] Yeni bbox'lar target'a ekleniyor
- [ ] Overlap kontrolü
- [ ] Visual test: Örnek copy-paste output

**Beklenen Etki:** +1-2% mAP (özellikle nadir sınıflar)

---

### Task 3.4: Transform Pipeline

**Dosya:** `src/augmentations/transforms.py`

**Açıklama:**
Albumentations tabanlı transform pipeline. Geometric ve photometric augmentations.

**Özellikler:**
- BBox-aware transforms (Albumentations)
- Geometric: Flip, Scale, Rotate, Perspective
- Photometric: HSV, Brightness, Contrast, Blur
- Train vs Val pipelines

**Acceptance Criteria:**
- [ ] `build_train_transforms(config)` fonksiyonu
- [ ] `build_val_transforms(config)` fonksiyonu
- [ ] Albumentations BboxParams ile bbox transform
- [ ] HSV, flip, scale, blur transforms
- [ ] Config'den probability değerleri okunuyor

**Beklenen Etki:** Genel robustness

---

## Task 4: Dataset

### Task 4.1: Enhanced COCO Dataset

**Dosya:** `src/dataset.py`

**Açıklama:**
COCO format veri yükler, augmentation pipeline'ı uygular, Mosaic/MixUp destekler.

**Özellikler:**
- COCO JSON parsing
- On-the-fly augmentation
- Mosaic integration (dataset-level)
- MixUp integration
- Multi-scale support
- Collate function

**Acceptance Criteria:**
- [ ] `COCODetectionDataset(root, transforms, mosaic_prob, mixup_prob)` class'ı
- [ ] COCO annotations doğru parse ediliyor
- [ ] `__getitem__` augmented (image, target) döndürüyor
- [ ] Mosaic: 4 image fetch ve combine
- [ ] MixUp: 2 image blend
- [ ] `collate_fn` batch oluşturuyor
- [ ] Multi-scale resize desteği

**Beklenen Etki:** Tüm augmentation'ların çalışması için temel

---

## Task 5: Evaluation

### Task 5.1: COCO mAP Evaluator

**Dosya:** `src/evaluation/coco_eval.py`

**Açıklama:**
pycocotools kullanarak standart COCO mAP hesaplama.

**Metrikler:**
- AP (mAP@50:95) - Ana metrik
- AP50 (mAP@50)
- AP75 (mAP@75)
- APs (small objects)
- APm (medium objects)
- APl (large objects)
- Per-class AP

**Acceptance Criteria:**
- [ ] `COCOEvaluator(dataset)` class'ı
- [ ] `evaluate(model, dataloader)` methodu
- [ ] pycocotools entegrasyonu
- [ ] Tüm COCO metrikler dönüyor
- [ ] Per-class AP opsiyonel
- [ ] Unit test: Bilinen predictions için mAP doğru

**Beklenen Etki:** Doğru model değerlendirme

---

## Task 6: Config Updates

### Task 6.1: Enhanced Configuration

**Dosya:** `config.py`

**Değişiklikler:**

```python
# MEVCUT TrainingConfig'e eklenecekler:
@dataclass
class TrainingConfig:
    # ... mevcut alanlar ...

    # SOTA Features (YENİ)
    use_ema: bool = True
    ema_decay: float = 0.9999
    use_llrd: bool = True
    llrd_decay: float = 0.9
    gradient_clip: float = 0.1
    gradient_accumulation: int = 1
    multi_scale: bool = True
    scale_min: int = 480
    scale_max: int = 800

# YENİ dataclass'lar:
@dataclass
class AugmentationConfig:
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.3
    copypaste_prob: float = 0.2
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flip_prob: float = 0.5
    scale: tuple = (0.5, 1.5)
    degrees: float = 0.0
    translate: float = 0.1
    shear: float = 0.0

@dataclass
class LossConfig:
    cls_loss: str = "focal"
    box_loss: str = "ciou"
    cls_weight: float = 1.0
    box_weight: float = 5.0
    dfl_weight: float = 1.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
```

**Acceptance Criteria:**
- [ ] TrainingConfig'e SOTA alanlar eklendi
- [ ] AugmentationConfig dataclass oluşturuldu
- [ ] LossConfig dataclass oluşturuldu
- [ ] MODEL_CONFIGS'e D-FINE eklendi
- [ ] YOLO-NAS kaldırıldı
- [ ] Handler'dan config'ler okunabiliyor

---

## Task 7: Base Trainer Updates

### Task 7.1: SOTA Base Trainer

**Dosya:** `trainers/base.py`

**Değişiklikler:**

```python
class BaseTrainer:
    def __init__(self, ...):
        # Mevcut init
        self.ema = None
        self.scaler = None

    # YENİ methodlar:
    def setup_ema(self): ...
    def setup_optimizer(self): ...  # LLRD support
    def setup_scheduler(self): ...
    def setup_amp(self): ...        # Mixed precision

    # GÜNCELLENMİŞ train loop:
    def train(self):
        self.setup_ema()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_amp()

        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)

            # EMA validation
            if self.ema:
                self.ema.apply_shadow(self.model)
            val_metrics = self.validate()
            if self.ema:
                self.ema.restore(self.model)

            # Checkpoint (EMA weights)
            ...
```

**Acceptance Criteria:**
- [ ] EMA setup ve integration
- [ ] LLRD optimizer setup
- [ ] Scheduler setup
- [ ] Mixed precision (AMP) setup
- [ ] Train loop EMA update içeriyor
- [ ] Validation EMA weights kullanıyor
- [ ] Checkpoint EMA weights kaydediyor

---

## Task 8: Model Trainers

### Task 8.1: RT-DETR Trainer (Rewrite)

**Dosya:** `trainers/rt_detr.py`

**Değişiklikler:**
- Enhanced dataset ile augmentation
- SOTA training loop (base'den inherit)
- COCO mAP evaluation
- Multi-scale training

**Acceptance Criteria:**
- [ ] HuggingFace RTDetrForObjectDetection yükleniyor
- [ ] Enhanced dataset kullanılıyor
- [ ] train_epoch: AMP, gradient clip, EMA update
- [ ] validate: COCO mAP evaluation
- [ ] Multi-scale training çalışıyor
- [ ] End-to-end test: Küçük dataset'te training çalışıyor

---

### Task 8.2: D-FINE Trainer (New)

**Dosya:** `trainers/dfine.py`

**Açıklama:**
D-FINE modeli için trainer. HuggingFace'ten yüklenir.

**Acceptance Criteria:**
- [ ] HuggingFace D-FINE model yükleniyor
- [ ] RT-DETR trainer ile aynı SOTA features
- [ ] DFL loss entegrasyonu
- [ ] End-to-end test: Küçük dataset'te training çalışıyor

---

## Task 9: Handler Updates

### Task 9.1: Handler D-FINE Support

**Dosya:** `handler.py`

**Değişiklikler:**
```python
# Ekleme:
elif model_type == "d-fine":
    result = train_dfine(...)

# Kaldırma:
# yolo-nas case'i silinecek
```

**Acceptance Criteria:**
- [ ] D-FINE dispatch eklendi
- [ ] YOLO-NAS kaldırıldı
- [ ] Config'den augmentation/loss config okunuyor

---

## Task 10: Dependencies & Docker

### Task 10.1: Requirements Update

**Dosya:** `requirements.txt`

**Yeni Dependencies:**
```
# Mevcut + eklemeler:
albumentations>=1.3.0
pycocotools>=2.0.6
```

**Acceptance Criteria:**
- [ ] Albumentations eklendi (bbox support için)
- [ ] pycocotools eklendi (COCO eval için)
- [ ] Tüm dependencies uyumlu

---

### Task 10.2: Dockerfile Update

**Dosya:** `Dockerfile`

**Acceptance Criteria:**
- [ ] Yeni dependencies install ediliyor
- [ ] src/ klasörü COPY ediliyor
- [ ] Build başarılı

---

## Implementation Order

```
Phase 1: Core Utilities (Bağımsız modüller)
├── 1.1 src/training/ema.py
├── 1.2 src/training/optimizer.py
└── 1.3 src/training/scheduler.py

Phase 2: Losses (Bağımsız modüller)
├── 2.1 src/losses/focal.py
├── 2.2 src/losses/iou.py
└── 2.3 src/losses/dfl.py

Phase 3: Augmentations (Bağımsız modüller)
├── 3.1 src/augmentations/mosaic.py
├── 3.2 src/augmentations/mixup.py
├── 3.3 src/augmentations/copypaste.py
└── 3.4 src/augmentations/transforms.py

Phase 4: Dataset & Evaluation
├── 4.1 src/dataset.py (augmentations'a bağlı)
└── 5.1 src/evaluation/coco_eval.py

Phase 5: Config & Base
├── 6.1 config.py
└── 7.1 trainers/base.py (Phase 1'e bağlı)

Phase 6: Trainers
├── 8.1 trainers/rt_detr.py (Phase 4-5'e bağlı)
└── 8.2 trainers/dfine.py

Phase 7: Integration
├── 9.1 handler.py
└── 10.1-2 requirements.txt, Dockerfile
```

---

## Expected Performance Improvements

| Özellik | Beklenen mAP Artışı |
|---------|---------------------|
| EMA | +0.5-2% |
| LLRD | +0.5-1% |
| Mosaic | +2-5% |
| MixUp | +0.5-1% |
| Copy-Paste | +1-2% |
| Multi-Scale | +1-2% |
| CIoU Loss | +0.5-1% |
| DFL | +0.5% |
| Focal Loss | +1-2% (imbalanced) |
| **Toplam** | **+5-15%** |

---

## Testing Strategy

### Unit Tests
- Her modül için isolated unit test
- Loss fonksiyonları: bilinen input/output
- Augmentations: visual output check
- Metrics: bilinen predictions için mAP

### Integration Tests
- Küçük synthetic dataset ile end-to-end training
- 5-10 epoch, mAP artıyor mu?
- Checkpoint save/load çalışıyor mu?
- Webhook'lar gönderiliyor mu?

### Production Tests
- Gerçek shelf detection dataset ile training
- RunPod'da çalışıyor mu?
- Memory/GPU kullanımı normal mi?

---

## Rollback Plan

Eğer yeni sistem sorun çıkarırsa:
1. `handler.py`'da eski trainer'lara dönüş
2. `config.py`'da SOTA features disable
3. Git tag ile eski versiyona dönüş

---

## Timeline Estimate

| Phase | Tasks | Complexity |
|-------|-------|------------|
| Phase 1 | EMA, LLRD, Scheduler | Medium |
| Phase 2 | Losses | Medium |
| Phase 3 | Augmentations | High |
| Phase 4-5 | Dataset, Eval, Config | Medium |
| Phase 6 | Trainers | High |
| Phase 7 | Integration | Low |

---

## Notes

- Tüm yeni kod `src/` altında, mevcut yapıyı bozmadan
- Mevcut API/DB/Storage değişmiyor
- Handler minimal değişiklik
- Geriye uyumluluk korunuyor (eski config'ler çalışır)
