# Augmentation Worker

Runpod serverless worker for image augmentation using BiRefNet segmentation and Albumentations transforms.

## Features

- **BiRefNet Segmentation**: High-quality background removal with GPU half-precision
- **3 Augmentation Pipelines**:
  - `light`: Subtle transforms for synthetic images
  - `heavy`: Aggressive transforms for more variation
  - `real`: Specialized transforms for real product photos
- **Idempotent Top-up**: Only generates missing images to reach target count
- **Background Composition**: Composite products onto backgrounds with shadow effects
- **Border Detection**: Automatically detects if images need resize
- **Thread Limiting**: `OMP_NUM_THREADS=1` for stability

## Input

```json
{
  "dataset_id": "uuid-of-dataset",
  "syn_target": 600,
  "real_target": 400,
  "backgrounds_path": "/path/to/backgrounds"
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_id | string | Yes* | - | Dataset ID for Supabase integration |
| dataset_path | string | Yes* | - | Local path (alternative to dataset_id) |
| syn_target | int | No | 600 | Target synthetic images per UPC |
| real_target | int | No | 400 | Target real images per UPC |
| backgrounds_path | string | No | - | Path to background images |

*Either `dataset_id` or `dataset_path` is required.

## Output

```json
{
  "status": "success",
  "type": "augmentation",
  "dataset_id": "uuid",
  "syn_produced": 1200,
  "real_produced": 800,
  "report": {
    "totals": {...},
    "items": [...]
  }
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| SUPABASE_URL | Supabase project URL |
| SUPABASE_SERVICE_KEY | Supabase service role key |
| CALLBACK_URL | Backend webhook URL |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires dataset_path)
python src/handler.py
```

## Docker Build

```bash
docker build -t augmentation-worker .
docker run --gpus all -e SUPABASE_URL=... -e SUPABASE_SERVICE_KEY=... augmentation-worker
```

## Based On

`Eski kodlar/final_augmentor_v3.py` - All optimizations preserved.
