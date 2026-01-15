# BuyBuddy API - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Debug: verify files are copied
RUN echo "=== Checking /app/src structure ===" && \
    ls -la /app/src/ && \
    echo "=== Checking auth directory ===" && \
    ls -la /app/src/auth/ && \
    echo "=== All files copied successfully ==="

# Set Python path and default port
ENV PYTHONPATH=/app/src
ENV PORT=8000

# Expose port (Railway overrides this with $PORT)
EXPOSE $PORT

# Change to src directory and run the application
WORKDIR /app/src
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
