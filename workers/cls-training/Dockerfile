FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY handler.py .

# Pre-download some models
RUN python -c "import timm; timm.create_model('efficientnet_b0', pretrained=True)"

CMD ["python", "-u", "handler.py"]
