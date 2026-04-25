# ---------------------------------------------------------------
# RunPod GPU Worker — Qwen2.5 + LLaMA-Factory fine-tuning
# ---------------------------------------------------------------

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy project FIRST ----
COPY . .

# ---- Upgrade pip ----
RUN pip install --no-cache-dir --upgrade pip

# ---- Install project dependencies (your pipeline, celery, etc.) ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Install GPU / LLM training stack ----
RUN pip install --no-cache-dir \
    "llamafactory[torch,metrics]" \
    deepspeed \
    bitsandbytes \
    peft \
    trl \
    wandb \
    accelerate

# ---- Make startup script executable ----
RUN chmod +x docker/runpod/start.sh

# ---- Workspace (persistent volume mount) ----
RUN mkdir -p /workspace

CMD ["bash", "docker/runpod/start.sh"]