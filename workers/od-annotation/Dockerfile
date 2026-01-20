# OD Annotation Worker - SOTA AI Auto-Annotation (Optimized)
# All models pre-downloaded for fast cold starts (~10-20s)
# Base: CUDA 12.1 + Python 3.11 for RunPod Serverless

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ============================================
# Stage 1: System Dependencies (rarely changes)
# ============================================
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
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

# Set working directory
WORKDIR /app

# ============================================
# Stage 2: Python Dependencies (changes occasionally)
# ============================================
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support + all requirements in one layer
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# ============================================
# Stage 3: ML Libraries from Source (changes rarely)
# ============================================
# Install Grounding DINO and SAM2 from source
RUN git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO && \
    cd /tmp/GroundingDINO && \
    pip install --no-build-isolation -e . && \
    rm -rf /tmp/GroundingDINO/.git && \
    pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git && \
    rm -rf /root/.cache/pip

# ============================================
# Stage 4: Model Weights (parallel download for speed)
# ============================================
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN mkdir -p /app/weights /app/huggingface_cache

# Download all model weights in parallel using background processes
RUN echo "Downloading all model weights in parallel..." && \
    wget -q -O /app/weights/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth & \
    PID1=$! && \
    wget -q -O /app/weights/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt & \
    PID2=$! && \
    wait $PID1 $PID2 && \
    echo "Direct downloads complete!"

# Pre-download HuggingFace models (Florence-2 + SAM HF)
# Using separate script file for cleaner build
COPY scripts/download_hf_models.py /tmp/download_hf_models.py
RUN python /tmp/download_hf_models.py && rm -rf /root/.cache/pip /tmp/download_hf_models.py

# ============================================
# Stage 5: Application Code (changes frequently - last for caching)
# ============================================
COPY . .

# Environment variables for model paths
ENV GROUNDING_DINO_WEIGHTS=/app/weights/groundingdino_swint_ogc.pth
ENV SAM2_WEIGHTS=/app/weights/sam2.1_hiera_large.pt

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
