# OD Training Worker - Dockerfile
# For training RT-DETR and D-FINE models with SOTA features
# Based on RunPod PyTorch template for compatibility

# PyTorch 2.8 + CUDA 12.8 - Supports ALL GPUs including RTX 5090 (Blackwell sm_120)
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# Note: torch is already installed in base image, but we upgrade other deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/training_output /tmp/checkpoints /tmp/logs /tmp/huggingface /tmp/torch

# Pre-download models for faster cold starts (optional, increases image size)
# RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('PekingU/rtdetr_r18vd')"

# Set permissions
RUN chmod -R 755 /app

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import torch; import transformers; print(f'GPU: {torch.cuda.is_available()}')" || exit 1

# Run the handler
CMD ["python", "-u", "handler.py"]
