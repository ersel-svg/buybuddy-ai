# bb-models

BuyBuddy AI shared model package for embedding extraction and training.

## Features

- **Multiple Model Families**: DINOv2, DINOv3, CLIP
- **Training Heads**: GeM Pooling, ArcFace, Projection heads
- **LLRD Support**: Layer-wise Learning Rate Decay
- **Checkpoint Management**: Automatic best/final tracking with retention policy
- **Config Presets**: Optimized hyperparameters per model

## Supported Models

### DINOv2 Family
| Model ID | Embedding Dim | Params | Notes |
|----------|---------------|--------|-------|
| `dinov2-small` | 384 | 22M | Fast, lightweight |
| `dinov2-base` | 768 | 86M | Balanced |
| `dinov2-large` | 1024 | 300M | High accuracy |

### DINOv3 Family
| Model ID | Embedding Dim | Params | Notes |
|----------|---------------|--------|-------|
| `dinov3-small` | 384 | 21M | Latest architecture |
| `dinov3-base` | 768 | 86M | **Recommended** |
| `dinov3-large` | 1024 | 300M | Best accuracy |

### CLIP Family
| Model ID | Embedding Dim | Params | Notes |
|----------|---------------|--------|-------|
| `clip-vit-l-14` | 1024 | 304M | High capacity |

## Installation

```bash
# From the monorepo root
pip install -e packages/bb-models

# With training dependencies
pip install -e "packages/bb-models[training]"
```

## Quick Start

### Embedding Extraction

```python
from bb_models import get_backbone

# Load a pretrained backbone
backbone = get_backbone("dinov2-base")

# Extract embeddings
embeddings = backbone(images)  # (B, 768)
```

### Custom Checkpoint

```python
backbone = get_backbone(
    "dinov2-base",
    checkpoint_url="https://storage.example.com/model.pth"
)
```

### Training Configuration

```python
from bb_models.configs import get_preset, merge_config

# Get optimized preset
config = get_preset("dinov3-base")

# Customize
config = merge_config(config, {
    "training": {"epochs": 50, "batch_size": 32}
})
```

### LLRD Optimizer

```python
from bb_models.utils import get_llrd_optimizer_params

params = get_llrd_optimizer_params(
    model,
    base_lr=3e-5,
    llrd_decay=0.9,
    weight_decay=0.01,
)
optimizer = torch.optim.AdamW(params)
```

### Training Heads

```python
from bb_models.heads import GeMPooling, EnhancedArcFaceLoss, ProjectionHead

# GeM pooling for spatial features
pool = GeMPooling(p=3.0)

# ArcFace classification head
arcface = EnhancedArcFaceLoss(
    in_features=768,
    num_classes=1000,
    scale=30.0,
    margin=0.30,
    label_smoothing=0.15,
)

# Projection head
proj = ProjectionHead(in_features=768, out_features=512)
```

### Checkpoint Management

```python
from bb_models.utils import CheckpointManager

manager = CheckpointManager(
    output_dir="./checkpoints",
    keep_best=True,
    keep_final=True,
    keep_last_n=3,
)

# Save checkpoint (auto-tracks best, handles cleanup)
manager.save(
    model=model,
    optimizer=optimizer,
    epoch=10,
    train_loss=1.23,
    val_loss=1.45,
    val_recall_at_1=0.85,
)

# Get best checkpoint
best_path = manager.get_best_checkpoint()
```

## DINOv3 Access

DINOv3 models require Meta license agreement:

1. Accept license at: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
2. Set `HF_TOKEN` environment variable

```bash
export HF_TOKEN=your_huggingface_token
```

## Project Structure

```
packages/bb-models/
├── src/bb_models/
│   ├── __init__.py
│   ├── base.py           # BaseBackbone ABC
│   ├── registry.py       # Model configs & factory
│   ├── backbones/
│   │   ├── dinov2.py
│   │   ├── dinov3.py
│   │   └── clip.py
│   ├── heads/
│   │   ├── gem_pooling.py
│   │   ├── arcface.py
│   │   └── projection.py
│   ├── utils/
│   │   ├── checkpoint.py
│   │   ├── llrd.py
│   │   └── preprocessing.py
│   └── configs/
│       └── presets.py
├── pyproject.toml
└── README.md
```

## License

MIT
