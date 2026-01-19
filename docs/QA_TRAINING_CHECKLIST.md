# Training System - Production QA Checklist

> **Tarih:** 2026-01-19
> **Amaç:** Training sisteminin uçtan uca production testi
> **Test Ortamı:** Gerçek veri, gerçek flow, gerçek çıktılar

---

## P0 - Kritik (Mutlaka Test Edilmeli)

### 1. Training Akışı (End-to-End)
- [ ] Training run oluşturma (tüm config'lerle)
- [ ] RunPod job başlatma
- [ ] Progress reporting (epoch, batch)
- [ ] Checkpoint upload (FP16, 42MB limit)
- [ ] Training tamamlanma (status: completed)
- [ ] Best checkpoint belirleme

### 2. Veri İşleme
- [ ] `training_images` URL formatı (YENİ)
- [ ] Image download ve cache
- [ ] Domain dağılımı (synthetic/real/augmented)
- [ ] Min samples per class filtresi
- [ ] Train/val/test split (product bazlı, leak yok)

### 3. Evaluation
- [ ] Test set üzerinde evaluation
- [ ] Recall@1, Recall@5, Recall@10, mAP
- [ ] Cross-domain evaluation (real→synth, synth→real)
- [ ] Per-category breakdown
- [ ] Hard case identification

### 4. Model Kayıt
- [ ] Checkpoint'ten model kaydetme
- [ ] trained_models tablosuna insert
- [ ] Test metrics kaydetme
- [ ] Model dropdown'da görünme (extraction page)

---

## P1 - Önemli

### 5. Checkpoint Yönetimi
- [ ] Her epoch'ta checkpoint kaydetme
- [ ] `is_best` flag doğru set edilmesi
- [ ] Storage'a upload (Supabase)
- [ ] Checkpoint silme (storage + DB)
- [ ] Cascade delete koruması (linked model varsa)

### 6. Training Config
- [ ] Model tipleri: dinov2-base, dinov2-large, dinov3-base
- [ ] Data sources: all_products, matched_products, dataset
- [ ] Image config: synthetic, real, augmented, cutout
- [ ] Frame selection: first, key_frames, interval
- [ ] Split ratios: 70/15/15 default

### 7. SOTA Features
- [ ] ArcFace loss
- [ ] Combined loss (ArcFace + Triplet)
- [ ] P-K sampling
- [ ] Early stopping
- [ ] Warmup + cosine annealing
- [ ] Mixed precision (FP16)
- [ ] LLRD (layer-wise learning rate decay)

### 8. UI/UX
- [ ] Training runs listesi (5s refresh)
- [ ] Status badges (running, completed, failed)
- [ ] Progress bar ve metrics
- [ ] Create dialog form validation
- [ ] Advanced config collapsible
- [ ] Delete confirmation (force dialog)

---

## P2 - İyi Olur

### 9. Resume/Cancel
- [ ] Training cancel (RunPod job cancel)
- [ ] Training resume (checkpoint'ten devam)
- [ ] Remaining epochs hesaplama

### 10. Error Handling
- [ ] Yetersiz ürün (< 10)
- [ ] Yetersiz görsel (< 100)
- [ ] Broken image URL'leri
- [ ] Network timeout recovery
- [ ] RunPod job failure

### 11. Metrics History
- [ ] Epoch bazlı metrics kaydetme
- [ ] Loss breakdown (arcface, triplet, domain)
- [ ] Learning rate tracking
- [ ] Chart gösterimi

### 12. Model Comparison
- [ ] 2-5 model karşılaştırma
- [ ] Metric alignment
- [ ] Visual comparison

---

## Test Senaryoları (Gerçek Veri ile)

| ID | Senaryo | Veri | Config | Beklenen Sonuç |
|----|---------|------|--------|----------------|
| S1 | Minimal | 10 ürün, 200 görsel | dinov2-base, 5 epoch | Başarılı tamamlanma, checkpoint var |
| S2 | Standard | 50 ürün, 1K görsel | dinov2-base, 10 epoch | R@1 > 80% |
| S3 | SOTA | 50 ürün, 1K görsel | Combined loss, P-K sampling | R@1 > 85% |
| S4 | Large Model | 100+ ürün | dinov2-large | Memory OK, yavaş ama başarılı |
| S5 | Mixed Domain | Real + Synthetic | Cross-domain eval enabled | Domain metrics mevcut |

---

## Deployment Checklist

### Worker Hazırlık
- [ ] Training worker Docker image build edildi
- [ ] Docker image RunPod'a push edildi
- [ ] RUNPOD_ENDPOINT_ID env variable ayarlandı
- [ ] Worker health check başarılı

### Infrastructure
- [ ] Supabase Storage bucket var (`checkpoints`)
- [ ] Supabase tablolar mevcut (training_runs, checkpoints, trained_models)
- [ ] API endpoint'leri çalışıyor (`/api/v1/training/*`)
- [ ] Frontend build başarılı
- [ ] Qdrant bağlantısı OK

### Environment Variables
```env
SUPABASE_URL=✓
SUPABASE_SERVICE_ROLE_KEY=✓
RUNPOD_API_KEY=✓
RUNPOD_ENDPOINT_ID=✓ (training worker)
HF_TOKEN=✓ (DINOv3 için)
```

---

## Test Sonuçları

### S1: Minimal Test
| Adım | Durum | Notlar |
|------|-------|--------|
| Run oluşturma | ⏳ | |
| Job başlatma | ⏳ | |
| Epoch 1 tamamlandı | ⏳ | |
| Checkpoint upload | ⏳ | |
| Training completed | ⏳ | |
| Model kayıt | ⏳ | |

### S2: Standard Test
| Adım | Durum | Notlar |
|------|-------|--------|
| TBD | ⏳ | |

---

## Bilinen Sorunlar ve Düzeltmeler

### Düzeltildi
1. **Checkpoint 413 Payload Too Large** - FP16 conversion ile 263MB → 42MB
2. **Cross-domain KeyError: 0** - Integer → string domain mapping
3. **Cascade delete** - trained_models kontrolü eklendi

### Bekleyen
1. **Trained model extraction** - Worker checkpoint yükleme desteği gerekiyor

---

## Notlar

_Test sırasında notlar buraya eklenecek_

