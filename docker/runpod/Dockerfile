# ---------------- CUDA + PyTorch base ----------------
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ---------------- System deps ----------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------- Python deps ----------------
# (your existing requirements)
COPY ladybug/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------- LLM / training deps ----------------
RUN pip install --no-cache-dir \
    "transformers>=4.49.0,<=4.57.1,!=4.52.0,!=4.57.0" \
    "peft>=0.14.0,<=0.17.1" \
    "datasets>=2.16.0,<=4.0.0" \
    "accelerate>=1.3.0,<=1.11.0" \
    scikit-learn \
    tqdm \
    "pandas>=2.0.0" \
    bitsandbytes \
    wandb

# ---------------- LLaMA Factory ----------------
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git \
    && cd LLaMA-Factory \
    && pip install -e .

# ---------------- Your code ----------------
COPY ladybug /app/ladybug

# ---------------- ENV ----------------
COPY docker/env/.env.gpu_worker /app/.env
ENV PYTHONPATH=/app
