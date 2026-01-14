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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download DINOv2 models to avoid cold start delay
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
    AutoModel.from_pretrained('facebook/dinov2-base')"

# Copy source code
COPY src/ ./src/

# Environment variables (set at runtime)
ENV SUPABASE_URL=""
ENV SUPABASE_SERVICE_ROLE_KEY=""
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""

# Run handler
CMD ["python", "-u", "src/handler.py"]
