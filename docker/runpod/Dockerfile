FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# ─────────────────────────────
# System dependencies
# ─────────────────────────────
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────
# Copy project
# ─────────────────────────────
COPY . .

# ─────────────────────────────
# Python tooling
# ─────────────────────────────
RUN pip install --no-cache-dir --upgrade pip

# ─────────────────────────────
# Install your project
# ─────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────
# ML / GPU stack
# ─────────────────────────────
RUN pip install --no-cache-dir \
    llamafactory[torch,metrics] \
    deepspeed \
    bitsandbytes \
    peft \
    trl \
    accelerate \
    wandb \
    mlflow \
    celery \
    redis \
    boto3

# ─────────────────────────────
# RunPod persistent storage
# ─────────────────────────────
RUN mkdir -p /workspace

