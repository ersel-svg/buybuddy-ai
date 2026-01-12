FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install core Python dependencies first (matching video-segmentation order)
RUN pip install --no-cache-dir \
    runpod \
    torch>=2.0.0 \
    torchvision \
    opencv-python-headless \
    pillow \
    numpy \
    tqdm \
    supabase>=2.0.0 \
    httpx>=0.25.0 \
    albumentations>=1.3.0

# Install BiRefNet/transformers dependencies separately (like SAM3 deps in video-segmentation)
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
