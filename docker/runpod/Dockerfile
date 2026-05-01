# ════════════════════════════════════════════════════════════════
#  Ladybug — RunPod GPU Worker
#  Base: RunPod PyTorch (CUDA 11.8, Python 3.10)
#  Runs: Celery GPU worker → LLaMA-Factory fine-tuning
# ════════════════════════════════════════════════════════════════

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Copy project ─────────────────────────────────────────────────
COPY . .

# ── Upgrade pip ──────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip

# ── Project dependencies (pipeline, celery, pydantic-settings…) ──
RUN pip install --no-cache-dir -r requirements.txt

# ── GPU / LLM training stack ─────────────────────────────────────
RUN pip install --no-cache-dir \
    "llamafactory[torch,metrics]" \
    deepspeed \
    bitsandbytes \
    peft \
    trl \
    accelerate \
    wandb

# ── Persistent workspace (network volume mounts here on RunPod) ──
RUN mkdir -p /workspace

# ── Startup script ───────────────────────────────────────────────
RUN chmod +x docker/runpod/start.sh

CMD ["bash", "docker/runpod/start.sh"]