FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM3 (Segment Anything 3)
# Note: Replace with actual SAM3 installation when available
# RUN pip install git+https://github.com/facebookresearch/sam3.git
RUN pip install segment-anything-2

# Copy source code
COPY src/ ./src/

# Environment variables (set at runtime)
ENV GEMINI_API_KEY=""
ENV HF_TOKEN=""
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""
ENV CALLBACK_URL=""

# Run handler
CMD ["python", "-u", "src/handler.py"]
