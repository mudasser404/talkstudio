# RunPod Serverless Dockerfile for F5-TTS Voice Cloning API
# Base image with CUDA support and PyTorch pre-installed
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/cache/huggingface
ENV TORCH_HOME=/runpod-volume/cache/torch
ENV TRANSFORMERS_CACHE=/runpod-volume/cache/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_runpod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_runpod.txt

# Copy F5-TTS source
COPY F5-TTS /app/F5-TTS

# Install F5-TTS from local source
RUN cd /app/F5-TTS && pip install --no-cache-dir -e .

# Copy handler and config files
COPY handler.py .
COPY config.py .

# Create cache directories
RUN mkdir -p /runpod-volume/cache/huggingface /runpod-volume/cache/torch /runpod-volume/cache/transformers

# Pre-download model during build (reduces cold start time significantly)
# Uncomment if you want to bake model into the image (increases image size but faster cold start)
# RUN python -c "from f5_tts.api import F5TTS; F5TTS(model='F5TTS_v1_Base', device='cpu')"

# Set the handler entrypoint
CMD ["python", "-u", "handler.py"]