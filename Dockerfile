# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Librosa/Audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ ./src/
COPY web/ ./web/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Expose API port
EXPOSE 8000

# Environment Defaults
ENV MODEL_PATH=/app/models/best_model.onnx
ENV SQLITE_DB_PATH=/app/logs/audit_trail.db

# Launch Uvicorn FastAPI Server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
