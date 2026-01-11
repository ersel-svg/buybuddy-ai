FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SAM3 from GitHub
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git

# SAM3 dependencies (explicit)
RUN pip install --no-cache-dir einops decord timm iopath hydra-core omegaconf pycocotools

# Copy source code
COPY src/ ./src/

# Set environment
ENV PYTHONUNBUFFERED=1

# Runpod handler
CMD ["python", "-u", "src/handler.py"]
