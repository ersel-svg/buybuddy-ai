# OD Annotation Worker - SOTA AI Auto-Annotation
# All models pre-downloaded for fast cold starts (~10-20s)
# Models: Grounding DINO, Florence-2, SAM (HF), SAM3
# NOTE: SAM3 weights downloaded at runtime (requires HF_TOKEN env var)

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Token for SAM3 model download (set at runtime, not build time)
ENV HF_TOKEN=""

# ============================================
# Stage 1: System Dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
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

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ============================================
# Stage 2: Python Dependencies
# ============================================
COPY requirements.txt .

# Install PyTorch (CUDA 12.4) + requirements
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

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
# Stage 4: Model Weights
# ============================================
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

RUN mkdir -p /app/weights /app/huggingface_cache

# Grounding DINO weights (~700MB)
RUN echo "=== Downloading Grounding DINO weights ===" && \
    wget --progress=bar:force -O /app/weights/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# HuggingFace models (Florence-2 ~3GB, SAM-HF ~2.5GB)
COPY scripts/download_hf_models.py /tmp/download_hf_models.py
RUN echo "=== Downloading HuggingFace models ===" && \
    python /tmp/download_hf_models.py && \
    rm -f /tmp/download_hf_models.py

# ============================================
# Stage 5: Application Code
# ============================================
COPY . .

ENV GROUNDING_DINO_WEIGHTS=/app/weights/groundingdino_swint_ogc.pth

CMD ["python", "-u", "handler.py"]
