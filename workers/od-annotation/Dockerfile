# OD Annotation Worker - SOTA AI Auto-Annotation
# Base: CUDA 12.1 + Python 3.11 for RunPod Serverless

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install Grounding DINO from source
# Use --no-build-isolation to use existing torch instead of trying to install it again
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO && \
    cd /tmp/GroundingDINO && \
    pip install --no-build-isolation -e . && \
    rm -rf /tmp/GroundingDINO/.git

# Install SAM 2 from source (no-build-isolation to use existing torch)
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git

# Download model weights during build for faster cold starts
RUN mkdir -p /app/weights

# Grounding DINO weights
RUN wget -q -O /app/weights/groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# SAM 2.1 weights (large model)
RUN wget -q -O /app/weights/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Copy application code
COPY . .

# Set environment variables for model paths
ENV GROUNDING_DINO_WEIGHTS=/app/weights/groundingdino_swint_ogc.pth
ENV SAM2_WEIGHTS=/app/weights/sam2.1_hiera_large.pt
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
