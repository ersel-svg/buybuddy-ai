# OD Annotation Worker - SOTA AI Auto-Annotation
# Optimized for faster builds - models downloaded at runtime
# Models: Grounding DINO, Florence-2, SAM2, SAM3, Roboflow (YOLO/RF-DETR)

# Use RunPod base image (same as od-training) - smaller and faster
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Token for SAM3 model download (set at runtime, not build time)
ENV HF_TOKEN=""
ENV HF_HOME=/tmp/huggingface_cache
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV TORCH_HOME=/tmp/torch

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
RUN pip install --no-cache-dir groundingdino-py

# SAM3 dependencies (MUST be installed before SAM3)
RUN pip install --no-cache-dir einops decord timm iopath hydra-core omegaconf pycocotools

# SAM3 (from official repo - PINNED to same commit as video-segmentation worker)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git@7b89b8fc3fa0ae8d09d9b17a284b5299e238b1b0

# ============================================
# Stage 4: Create cache directories
# ============================================
# Models are downloaded at runtime for smaller image size
# First cold start will be slower but subsequent runs use persistent cache
RUN mkdir -p /tmp/huggingface_cache /tmp/torch /tmp/roboflow_models

# ============================================
# Stage 5: Application Code
# ============================================
COPY . .

CMD ["python", "-u", "handler.py"]
