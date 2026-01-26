# Unified Inference Worker
# Uses same PyTorch base image as od-annotation for GPU compatibility
# CUDA 12.4 works with RTX 4090, L4, A100, H100, etc.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch

WORKDIR /app

# Install system dependencies (same as od-annotation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies (torch/torchvision from base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Create cache directories
RUN mkdir -p /tmp/checkpoints /tmp/huggingface /tmp/torch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Start handler
CMD ["python", "-u", "handler.py"]
