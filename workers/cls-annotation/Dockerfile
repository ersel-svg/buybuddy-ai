FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     libgl1-mesa-glx     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Set environment variables
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache

# Preload models (optional - makes cold start faster but image bigger)
# RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', 'openai')"

CMD ["python", "-u", "handler.py"]
