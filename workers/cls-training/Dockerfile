# Use RunPod's official image - optimized for their GPU infrastructure (including RTX 5090)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY handler.py .

# Pre-download commonly used models (reduces cold start time)
RUN python -c "\
import timm; \
timm.create_model('efficientnet_b0', pretrained=True); \
timm.create_model('efficientnet_b2', pretrained=True); \
timm.create_model('convnext_tiny', pretrained=True); \
timm.create_model('vit_small_patch16_224', pretrained=True); \
print('Models pre-downloaded successfully')"

# Verify imports work
RUN python -c "from src.models import MODEL_REGISTRY; from src.losses import AVAILABLE_LOSSES; print(f'Ready: {len(MODEL_REGISTRY)} models, {len(AVAILABLE_LOSSES)} losses')"

CMD ["python", "-u", "handler.py"]
