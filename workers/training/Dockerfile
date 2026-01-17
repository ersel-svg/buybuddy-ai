# Official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: Build from monorepo root: docker build -f workers/training/Dockerfile .
COPY workers/training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install bb-models package
COPY packages/bb-models /app/bb-models
RUN pip install /app/bb-models[training]

# Pre-download default model (DINOv2 Base)
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
    AutoModel.from_pretrained('facebook/dinov2-base')"

# Copy source code
COPY workers/training/src/ ./src/

# Environment variables (set at runtime)
ENV SUPABASE_URL=""
ENV SUPABASE_SERVICE_ROLE_KEY=""
ENV HF_TOKEN=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV S3_BUCKET=""

# Run handler
CMD ["python", "-u", "src/handler.py"]
