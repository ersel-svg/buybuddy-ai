# OD Annotation Worker - SOTA AI Auto-Annotation
# All models pre-downloaded for fast cold starts (~10-20s)
# Models: Grounding DINO, Florence-2, SAM (HF), SAM3
# NOTE: SAM3 weights downloaded at runtime (requires HF_TOKEN env var)

# Use PyTorch base image (same as video-segmentation) for proper torchvision.ops support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Token for SAM3 model download (set at runtime, not build time)
ENV HF_TOKEN=""

WORKDIR /app

# ============================================
# Stage 1: System Dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================
# Stage 2: Python Dependencies
# ============================================
COPY requirements.txt .

# Install requirements (PyTorch already in base image)
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: ML Libraries
# ============================================
# Grounding DINO (pip package - pre-built, no CUDA compile)
RUN pip install groundingdino-py

# SAM3 dependencies (MUST be installed before SAM3)
RUN pip install --no-cache-dir einops decord timm iopath hydra-core omegaconf pycocotools

# SAM3 (from official repo - PINNED to same commit as video-segmentation worker)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git@7b89b8fc3fa0ae8d09d9b17a284b5299e238b1b0

# ============================================
# Stage 4: Model Weights (HuggingFace cache)
# ============================================
# All models load from HuggingFace at runtime or are pre-cached here
# - Grounding DINO: IDEA-Research/grounding-dino-base
# - Florence-2: microsoft/Florence-2-large (pre-downloaded)
# - SAM2: facebook/sam-vit-huge (pre-downloaded)
# - SAM3: downloads at runtime (requires HF_TOKEN)

ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

RUN mkdir -p /app/huggingface_cache

# Pre-download HuggingFace models for faster cold starts
COPY scripts/download_hf_models.py /tmp/download_hf_models.py
RUN echo "=== Downloading HuggingFace models ===" && \
    python /tmp/download_hf_models.py && \
    rm -f /tmp/download_hf_models.py

# ============================================
# Stage 5: Application Code
# ============================================
COPY . .

CMD ["python", "-u", "handler.py"]
