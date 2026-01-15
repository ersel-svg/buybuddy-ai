# Official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install system dependencies (git needed for pip install from GitHub)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM3 dependencies first
RUN pip install --no-cache-dir einops decord timm iopath hydra-core omegaconf pycocotools

# Install SAM3 (Segment Anything 3) from GitHub - pinned to working commit (CUDA 12.4 compatible)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git@7b89b8fc3fa0ae8d09d9b17a284b5299e238b1b0

# Copy source code
COPY src/ ./src/

# Environment variables (set at runtime)
ENV GEMINI_API_KEY=""
ENV HF_TOKEN=""
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""

# CUDA memory optimization - reduces fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default max frames (can be overridden)
ENV MAX_FRAMES=100

# Run handler
CMD ["python", "-u", "src/handler.py"]
