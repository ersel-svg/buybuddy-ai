# Official PyTorch image with CUDA 12.4 (same as video-segmentation)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies (ignore NVIDIA repo errors during mirror sync)
RUN apt-get update --allow-releaseinfo-change || true && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - matching tested pod environment
# Pod: Python 3.11, PyTorch 2.5.1, CUDA 12.4, NumPy 1.26.4
RUN pip install --no-cache-dir \
    runpod \
    opencv-python-headless \
    pillow \
    tqdm \
    supabase>=2.0.0 \
    httpx>=0.25.0 \
    albumentations>=2.0.0

# Install BiRefNet/transformers dependencies
# Versions matching tested pod: transformers 4.57.3, timm 1.0.24
RUN pip install --no-cache-dir einops kornia timm transformers

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
