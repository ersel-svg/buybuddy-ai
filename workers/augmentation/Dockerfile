FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ensure numpy is properly installed and compatible with PyTorch
RUN pip install --no-cache-dir --force-reinstall numpy

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Reinstall numpy after all deps to ensure compatibility
RUN pip install --no-cache-dir --force-reinstall numpy

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
