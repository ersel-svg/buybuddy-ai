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

# Install SAM3 (Segment Anything 3) from GitHub
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git

# Copy source code
COPY src/ ./src/

# Environment variables (set at runtime)
ENV GEMINI_API_KEY=""
ENV HF_TOKEN=""
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""

# Run handler
CMD ["python", "-u", "src/handler.py"]
