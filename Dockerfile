# AI Video Transcriber Docker image: Python matches the recommended local environment (3.12), with requirements.txt-compatible dependencies.
FROM python:3.12-slim-bookworm

WORKDIR /app

# System dependencies (FFmpeg for URL downloads and local upload transcoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first, then install from requirements.txt just like a fresh local environment.
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12

# CUDA libraries installed from pip are used by faster-whisper/ctranslate2.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# Copy project files
COPY . .

# Create temp and runtime cache directories
RUN mkdir -p temp .cache/huggingface .cache/yt-dlp

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8099
ENV WHISPER_MODEL_SIZE=base
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8
ENV WHISPER_BATCH_SIZE=8
ENV MAX_PARALLEL_LLM_REQUESTS=4
ENV XDG_CACHE_HOME=/app/.cache
ENV HF_HOME=/app/.cache/huggingface

# Expose port
EXPOSE 8099

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8099/ || exit 1

# Start command
CMD ["python3", "start.py", "--prod"]
