FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

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
