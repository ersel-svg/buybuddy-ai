# OD Training System - Comprehensive Test Plan

## Executive Summary
Bu test planı, OD Training sisteminin tüm özelliklerini, konfigürasyonlarını ve entegrasyon noktalarını kapsamlı şekilde test etmek için hazırlanmıştır.

---

## Test Kategorileri

### 1. MODEL TESTLERİ

#### 1.1 RT-DETR Model Tests
| Test ID | Test | Dataset Size | Expected Result |
|---------|------|--------------|-----------------|
| M-001 | RT-DETR Small (s) | 100 images | Training completes, mAP > 0 |
| M-002 | RT-DETR Medium (m) | 100 images | Training completes, mAP > 0 |
| M-003 | RT-DETR Large (l) | 100 images | Training completes, mAP > 0 |

#### 1.2 D-FINE Model Tests
| Test ID | Test | Dataset Size | Expected Result |
|---------|------|--------------|-----------------|
| M-004 | D-FINE Small (s) | 100 images | Training completes, mAP > 0 |
| M-005 | D-FINE Medium (m) | 100 images | Training completes, mAP > 0 |
| M-006 | D-FINE Large (l) | 100 images | Training completes, mAP > 0 |
| M-007 | D-FINE XLarge (x) | 100 images | Training completes, mAP > 0 |

---

### 2. AUGMENTATION TESTLERİ

#### 2.1 Preset Tests
| Test ID | Preset | Expected Behavior |
|---------|--------|-------------------|
| A-001 | sota-v2 | Mosaic9, MixUp, CopyPaste aktif |
| A-002 | sota | Mosaic, MixUp aktif |
| A-003 | heavy | Aggressive augmentations |
| A-004 | medium | Balanced augmentations |
| A-005 | light | Minimal augmentations |
| A-006 | none | No augmentations |

#### 2.2 Multi-Image Augmentation Tests
| Test ID | Augmentation | Config | Expected |
|---------|--------------|--------|----------|
| A-007 | Mosaic | prob=0.5 | 4-image grid |
| A-008 | Mosaic9 | prob=0.3 | 9-image grid |
| A-009 | MixUp | prob=0.3, alpha=8.0 | Alpha blending |
| A-010 | CutMix | prob=0.3 | Region cut+paste |
| A-011 | CopyPaste | prob=0.3 | Object copying |

#### 2.3 Custom Augmentation Tests
| Test ID | Test | Config |
|---------|------|--------|
| A-012 | Single augmentation | Only horizontal_flip |
| A-013 | Multiple augs | flip + rotate + blur |
| A-014 | All augs disabled | Custom with all off |
| A-015 | Max probability | All augs at prob=1.0 |

---

### 3. TRAINING CONFIG TESTLERİ

#### 3.1 Batch Size Tests
| Test ID | Batch Size | Dataset | Expected |
|---------|------------|---------|----------|
| T-001 | 4 | 100 images | 25 batches/epoch |
| T-002 | 8 | 100 images | 12-13 batches/epoch |
| T-003 | 16 | 100 images | 6-7 batches/epoch |
| T-004 | 32 | 100 images | 3-4 batches/epoch |
| T-005 | 64 | 100 images | GPU OOM check |

#### 3.2 Learning Rate Tests
| Test ID | Base LR | Expected |
|---------|---------|----------|
| T-006 | 0.00001 | Slow convergence |
| T-007 | 0.0001 | Normal (default) |
| T-008 | 0.001 | Fast convergence |

#### 3.3 Epoch Tests
| Test ID | Epochs | Expected |
|---------|--------|----------|
| T-009 | 5 | Quick validation |
| T-010 | 50 | Medium training |
| T-011 | 100 | Full training |

#### 3.4 Image Size Tests
| Test ID | Size | Expected |
|---------|------|----------|
| T-012 | 320 | Fast, lower accuracy |
| T-013 | 640 | Default, balanced |
| T-014 | 1280 | Slow, higher accuracy |

---

### 4. SOTA FEATURE TESTLERİ

#### 4.1 EMA Tests
| Test ID | EMA | Decay | Expected |
|---------|-----|-------|----------|
| S-001 | Enabled | 0.9999 | Stable training |
| S-002 | Disabled | - | Baseline comparison |
| S-003 | Enabled | 0.999 | Faster EMA updates |

#### 4.2 LLRD Tests
| Test ID | LLRD | Decay | Expected |
|---------|------|-------|----------|
| S-004 | Enabled | 0.9 | Layer-wise decay |
| S-005 | Disabled | - | Uniform LR |
| S-006 | Enabled | 0.8 | Aggressive decay |

#### 4.3 Mixed Precision Tests
| Test ID | AMP | Expected |
|---------|-----|----------|
| S-007 | Enabled | FP16, faster training |
| S-008 | Disabled | FP32, baseline |

#### 4.4 Scheduler Tests
| Test ID | Scheduler | Expected |
|---------|-----------|----------|
| S-009 | cosine | Warmup + cosine decay |
| S-010 | step | Step decay at milestones |
| S-011 | linear | Linear decay |
| S-012 | onecycle | One cycle scheduling |

---

### 5. DATASET FORMAT TESTLERİ

#### 5.1 COCO Format Tests
| Test ID | Test | Expected |
|---------|------|----------|
| D-001 | Standard COCO | images/train, images/val |
| D-002 | COCO without val | Use train for validation |
| D-003 | COCO with instances_train.json | Alternative naming |
| D-004 | COCO with num_classes in config | Override from annotations |

#### 5.2 YOLO Format Tests
| Test ID | Test | Expected |
|---------|------|----------|
| D-005 | Standard YOLO | Convert to COCO |
| D-006 | YOLO with data.yaml | Read class names |
| D-007 | YOLO without data.yaml | Default classes |

#### 5.3 Dataset Size Tests
| Test ID | Size | Expected |
|---------|------|----------|
| D-008 | 10 images | Small dataset handling |
| D-009 | 100 images | Standard test |
| D-010 | 1000 images | Medium dataset |
| D-011 | 10000 images | Large dataset |

---

### 6. FRONTEND ENTEGRASYON TESTLERİ

#### 6.1 Training Wizard Tests
| Test ID | Step | Test |
|---------|------|------|
| F-001 | Dataset | Select existing dataset |
| F-002 | Preprocess | Configure preprocessing |
| F-003 | Offline Aug | Skip (optional) |
| F-004 | Online Aug | Select SOTA-v2 preset |
| F-005 | Model | Select RT-DETR Large |
| F-006 | Hyperparams | Set epochs=10, batch=8 |
| F-007 | Review | Verify all settings |
| F-008 | Submit | Create training run |

#### 6.2 Progress Tracking Tests (CRITICAL)
| Test ID | Test | Expected |
|---------|------|----------|
| F-009 | Training list page | Shows all training runs |
| F-010 | Training detail page | Shows current status |
| F-011 | Progress updates | Webhook updates reflected |
| F-012 | Metrics display | Loss/mAP charts |
| F-013 | Completion notification | Toast on complete |

#### 6.3 API Integration Tests
| Test ID | Endpoint | Test |
|---------|----------|------|
| F-014 | POST /training | Create training run |
| F-015 | GET /training | List with filters |
| F-016 | GET /training/{id} | Get single run |
| F-017 | POST /training/webhook | Receive progress |
| F-018 | GET /training/{id}/metrics | Get metrics history |

---

### 7. WEBHOOK & PROGRESS TESTLERİ (CRITICAL - User Issue)

#### 7.1 Webhook Flow Tests
| Test ID | Test | Expected |
|---------|------|----------|
| W-001 | Webhook receives status=started | DB updated, UI shows "Starting" |
| W-002 | Webhook receives status=downloading | DB updated, UI shows "Downloading" |
| W-003 | Webhook receives status=training | DB updated, progress visible |
| W-004 | Webhook per epoch | Progress increments |
| W-005 | Webhook receives status=completed | Model URL stored |
| W-006 | Webhook receives status=failed | Error message shown |

#### 7.2 Webhook Authentication Tests
| Test ID | Test | Expected |
|---------|------|----------|
| W-007 | Webhook with Supabase key | 200 OK |
| W-008 | Webhook without auth | 401 Unauthorized |
| W-009 | Webhook with wrong key | 401 Unauthorized |

#### 7.3 Database Update Tests
| Test ID | Test | Expected |
|---------|------|----------|
| W-010 | training_runs.status update | Correct status |
| W-011 | training_runs.progress update | 0-100% |
| W-012 | training_runs.metrics_history | Array appended |
| W-013 | training_runs.model_url | Set on completion |

---

### 8. ERROR HANDLING TESTLERİ

#### 8.1 Input Validation Tests
| Test ID | Test | Expected |
|---------|------|----------|
| E-001 | Missing dataset_url | Error: dataset_url required |
| E-002 | Invalid model_type | Error: unsupported model |
| E-003 | Invalid batch_size | Error or fallback |
| E-004 | epochs=0 | Error: epochs must be > 0 |

#### 8.2 Runtime Error Tests
| Test ID | Test | Expected |
|---------|------|----------|
| E-005 | GPU OOM | Graceful failure, error message |
| E-006 | Dataset download fail | Error: download failed |
| E-007 | Invalid annotations | Error: parse failed |
| E-008 | num_classes=0 | Auto-detect from annotations |

#### 8.3 Recovery Tests
| Test ID | Test | Expected |
|---------|------|----------|
| E-009 | Model upload timeout | Retry with backoff |
| E-010 | Webhook failure | Continue training |
| E-011 | Early stopping | Clean exit |

---

### 9. EDGE CASE TESTLERİ

| Test ID | Test | Expected |
|---------|------|----------|
| EC-001 | 1 class dataset | Training works |
| EC-002 | 100+ classes | Training works |
| EC-003 | Very small images (32x32) | Resize + training |
| EC-004 | Very large images (4K) | Resize + training |
| EC-005 | Empty bboxes in some images | Skip/handle gracefully |
| EC-006 | Unicode class names | Works correctly |
| EC-007 | Mixed image formats (jpg, png, webp) | All supported |

---

### 10. PERFORMANCE TESTLERİ

| Test ID | Test | Metric |
|---------|------|--------|
| P-001 | Training speed (images/sec) | Benchmark |
| P-002 | GPU memory usage | Track peak |
| P-003 | Model upload time (650MB) | < 5 minutes |
| P-004 | Total training time (100 epochs) | Benchmark |

---

## Test Execution Priority

### Phase 1: Critical Path (Day 1)
1. W-001 to W-006: Webhook flow (USER ISSUE)
2. F-009 to F-013: Progress tracking (USER ISSUE)
3. M-002: RT-DETR Medium baseline
4. D-001: Standard COCO format
5. T-007: Default learning rate

### Phase 2: Core Features (Day 2)
1. All Model tests (M-001 to M-007)
2. Augmentation presets (A-001 to A-006)
3. SOTA features (S-001 to S-012)
4. Dataset formats (D-001 to D-007)

### Phase 3: Edge Cases (Day 3)
1. Error handling (E-001 to E-011)
2. Edge cases (EC-001 to EC-007)
3. Performance tests (P-001 to P-004)
4. Full frontend flow (F-001 to F-018)

---

## Test Data Requirements

### Datasets Needed:
1. **Slot Detection V1** (existing) - 100 images with 7 classes
2. **Single class dataset** - For EC-001
3. **Many class dataset** (50+ classes) - For EC-002
4. **YOLO format dataset** - For D-005 to D-007
5. **Problematic dataset** (empty bboxes, bad images) - For error tests

---

## Success Criteria

- [ ] All webhook tests pass (W-*)
- [ ] Frontend progress tracking works (F-009 to F-013)
- [ ] All model types train successfully (M-*)
- [ ] All augmentation presets work (A-001 to A-006)
- [ ] All SOTA features work (S-*)
- [ ] Both dataset formats work (D-*)
- [ ] Error handling graceful (E-*)
- [ ] Edge cases handled (EC-*)

---

## Known Issues to Fix

1. **Webhook 401** - Authentication headers added, needs deploy
2. **Batch logging** - Only shows batch 0, fix applied
3. **Deprecation warnings** - torch.amp fixes applied
4. **LR scheduler warning** - Fix applied
5. **Frontend progress** - Need to verify webhook → DB → UI flow

---

## Test Environment

- **RunPod Pod**: Direct testing with SSH
- **RunPod Serverless**: Production-like testing
- **Local API**: Development testing
- **Production API**: Final validation

