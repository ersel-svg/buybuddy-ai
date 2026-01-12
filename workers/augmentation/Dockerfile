FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies (ignore NVIDIA repo errors during mirror sync)
RUN apt-get update --allow-releaseinfo-change || true && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies WITHOUT touching torch/numpy (already in base image)
# Base image has: PyTorch 2.2.0 + NumPy 1.x (compatible)
RUN pip install --no-cache-dir \
    runpod \
    opencv-python-headless \
    pillow \
    tqdm \
    supabase>=2.0.0 \
    httpx>=0.25.0 \
    albumentations>=1.3.0

# Install BiRefNet/transformers dependencies separately (like SAM3 in video-segmentation)
# IMPORTANT: transformers pulls numpy as dependency, which may upgrade to 2.x
# We must pin numpy<2.0 to maintain PyTorch compatibility
RUN pip install --no-cache-dir einops kornia timm transformers && \
    pip install --no-cache-dir "numpy<2.0" --force-reinstall

# Copy source code
COPY src/ ./src/

# Set environment variables for thread limiting
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Supabase configuration (set at runtime)
# ENV SUPABASE_URL=
# ENV SUPABASE_SERVICE_KEY=
# ENV CALLBACK_URL=

WORKDIR /app/src
CMD ["python", "handler.py"]
