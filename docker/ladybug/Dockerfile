# CUDA base — needed for llamafactory GPU training
FROM hiyouga/llamafactory:latest

WORKDIR /app

# Install system dependencies your ladybug code needs
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install your requirements
COPY ladybug/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your ladybug source code
COPY ladybug /app/ladybug

# Copy the env file
COPY docker/env/.env.gpu_worker /app/.env

ENV PYTHONPATH=/app