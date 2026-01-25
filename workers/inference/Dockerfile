# Unified Inference Worker
# Uses RunPod PyTorch base image for fast builds and GPU compatibility
# Supports ALL GPUs: RTX 5090 (Blackwell), RTX 4090, L4, A100, H100, etc.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies (torch/torchvision from base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Create cache directory
RUN mkdir -p /tmp/checkpoints /tmp/huggingface /tmp/torch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Start handler
CMD ["python", "-u", "handler.py"]
