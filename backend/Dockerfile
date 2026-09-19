# Backend Dockerfile for Render deployment
# Docker context: repo root (.)

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies (build tools + curl for the model fetch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-fetch the ONNX embedding model at build time. Production then never
# touches the network on start-up, and every image carries the identical
# model files (the Hub cache can otherwise be left incomplete — see
# scripts/fetch_embedding_model.sh).
ARG EMBEDDING_MODEL_URL="https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"
RUN mkdir -p /model \
    && curl -fsSL --retry 3 -o /tmp/model.tar.gz "${EMBEDDING_MODEL_URL}" \
    && tar xzf /tmp/model.tar.gz -C /model --strip-components=1 --exclude='._*' 2>/dev/null \
    && rm /tmp/model.tar.gz

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ ./backend/

# Copy ML models (backend references ml/models relative to repo root)
COPY ml/models/ ./ml/models/

# Copy the pre-fetched ONNX embedding model to where app/ml/embeddings.py looks
COPY --from=builder /model ./backend/models/all-MiniLM-L6-v2-onnx

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check — the embedding model is already on disk, so start-up is fast
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
