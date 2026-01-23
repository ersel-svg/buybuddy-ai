# Classification Module - Implementation Plan

## Current Status

### COMPLETED
- [x] Database migrations (cls_images, cls_datasets, cls_classes, cls_labels, etc.)
- [x] Backend API endpoints (images, datasets, classes, labels, labeling, training, models, ai)
- [x] Frontend pages (dashboard, datasets, labeling with AI suggest)
- [x] API client methods for frontend

### REMAINING
- [ ] CLS_ANNOTATION worker (CLIP + SigLIP)
- [ ] CLS_TRAINING worker deploy
- [ ] E2E tests

---

## Task 1: CLS_ANNOTATION Worker

**Purpose:** AI labeling assistance using zero-shot classification

### Models
1. **CLIP** (OpenAI) - `open_clip` library
   - ViT-B/32, ViT-L/14
   - Zero-shot: image + class names → similarity scores

2. **SigLIP** (Google) - `transformers` library
   - Better than CLIP for some tasks
   - Same interface: image + text → similarity

### Worker Input/Output

```python
# Input
{
    "task": "classify",
    "model": "clip",  # or "siglip"
    "images": [
        {"id": "uuid", "url": "https://..."}
    ],
    "classes": ["class1", "class2", "class3"],  # Zero-shot class names
    "top_k": 3,  # Return top N predictions
    "threshold": 0.1  # Min confidence
}

# Output
{
    "results": [
        {
            "id": "uuid",
            "predictions": [
                {"class": "class1", "confidence": 0.85},
                {"class": "class2", "confidence": 0.12}
            ]
        }
    ]
}
```

### Files to Create
1. `/workspace/cls_annotation_worker.py` - Main worker handler
2. `/workspace/cls_models.py` - CLIP & SigLIP model loading

### Implementation Steps
1. SSH to pod
2. Create worker files
3. Test locally with sample images
4. Package as RunPod serverless endpoint

---

## Task 2: CLS_TRAINING Worker

**Purpose:** Train custom classification models

### Models (Already tested)
- ViT (vit_base_patch16_224)
- ConvNeXt (convnext_tiny)
- EfficientNet (efficientnet_b0)
- Swin (swin_tiny_patch4_window7_224)

### Worker Input/Output

```python
# Input
{
    "task": "train",
    "config": {
        "model_name": "vit_base_patch16_224",
        "num_classes": 4,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "augmentation": "medium"
    },
    "dataset": {
        "train_urls": [...],
        "val_urls": [...],
        "class_names": ["class1", "class2", ...]
    },
    "output_path": "s3://bucket/model.pt"
}

# Output
{
    "status": "completed",
    "metrics": {
        "train_loss": 0.15,
        "val_accuracy": 0.92,
        "best_epoch": 8
    },
    "model_url": "s3://bucket/model.pt"
}
```

### Files (Already exist)
- Training code tested on pod
- Need to package as RunPod serverless

### Implementation Steps
1. Review existing training code
2. Create proper RunPod handler
3. Add model upload to S3/Supabase
4. Deploy as serverless endpoint

---

## Task 3: E2E Tests

### Test 1: Manual Workflow (No AI)
```
1. Health check
2. Upload 10 test images
3. Create dataset
4. Create 4 classes (unique names)
5. Add images to dataset
6. Label images manually
7. Auto-split (train/val/test)
8. Create dataset version
9. Verify stats
```

### Test 2: AI Labeling Workflow
```
1. Create dataset with classes
2. Upload images
3. Call AI predict endpoint
4. Verify predictions returned
5. Auto-apply labels
6. Verify labels saved
```

### Test 3: Training Workflow
```
1. Prepare labeled dataset
2. Start training job
3. Poll job status
4. Verify model saved
5. Test inference with trained model
```

---

## RunPod Configuration

### New Environment Variables Needed
```env
RUNPOD_ENDPOINT_CLS_ANNOTATION=xxx  # New endpoint
RUNPOD_ENDPOINT_CLS_TRAINING=xxx    # New endpoint
```

### Endpoint Types (in runpod.py)
```python
class EndpointType(str, Enum):
    OD_ANNOTATION = "od_annotation"
    OD_TRAINING = "od_training"
    CLS_ANNOTATION = "cls_annotation"  # NEW
    CLS_TRAINING = "cls_training"      # NEW
```

---

## Execution Order

1. **CLS_ANNOTATION worker** (CLIP + SigLIP)
   - Create worker code on pod
   - Test locally
   - Deploy to RunPod
   - Update API to use new endpoint
   - Test AI labeling in frontend

2. **CLS_TRAINING worker**
   - Package existing code
   - Deploy to RunPod
   - Test training flow

3. **E2E Tests**
   - Fix test script (unique class names)
   - Run manual workflow test
   - Run AI labeling test
   - Run training test

---

## Time Estimate

| Task | Complexity |
|------|-----------|
| CLS_ANNOTATION worker | Medium |
| CLS_TRAINING worker | Easy (code exists) |
| E2E Tests | Easy |

Total: ~2-3 hours of focused work
