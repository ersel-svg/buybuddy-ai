# Buybuddy AI Platform

Internal tool for processing product videos into AI training data and product directory.

## Quick Start

### 1. Clone and Setup

```bash
cd buybuddy-ai
```

### 2. Read Context Files

Before starting development, read these files:
- `CONTEXT.md` - Full project context, APIs, credentials, schema
- `PROJECT_PLAN.md` - Detailed roadmap and tasks
- `.cursorrules` - AI assistant instructions

### 3. Development

**Worker (Runpod Serverless):**
```bash
cd worker
docker build -t buybuddy-worker .
docker run --gpus all \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  buybuddy-worker
```

**UI (NiceGUI):**
```bash
cd app
pip install -r requirements.txt
python main.py
```

## Project Structure

```
buybuddy-ai/
├── CONTEXT.md          # Full project context
├── PROJECT_PLAN.md     # Roadmap and tasks
├── .cursorrules        # AI assistant rules
│
├── worker/             # Runpod Serverless Worker
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── handler.py  # Runpod entrypoint
│       ├── pipeline.py # Main pipeline
│       └── config.py
│
├── app/                # NiceGUI Frontend (TODO)
│
├── shared/             # Shared code (TODO)
│
└── notebooks/          # Reference notebooks
    └── pipeline_prototype.ipynb
```

## Pipeline Flow

```
Video URL → Download → Gemini (metadata) → SAM3 (segment) → 518x518 Frames
```

## Key Technologies

- **SAM3**: Text-prompted video segmentation
- **Gemini Flash 2.5**: Metadata extraction
- **Runpod Serverless**: GPU inference
- **Supabase**: Database + Storage
- **NiceGUI**: Python-based UI
