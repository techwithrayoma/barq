#!/bin/bash
# ---------------------------------------------------------------
# RunPod GPU Worker Startup Script
# Connects to your RabbitMQ broker and listens ONLY on gpu queue
# ---------------------------------------------------------------
set -e

echo "=============================================="
echo "  RunPod GPU Worker — Ladybug LLM Training"
echo "=============================================="

# ---- Show GPU info so you can confirm the pod is correct ----
echo ""
echo ">>> GPU Info:"
nvidia-smi
echo ""

# ---- Confirm env vars are present ----
echo ">>> Broker  : $CELERY_BROKER_URL"
echo ">>> Backend : $CELERY_RESULT_BACKEND"
echo ">>> Project : /app"
echo ""

# ---- Give broker 5s to be fully reachable after pod start ----
echo ">>> Waiting 5s for broker to be reachable..."
sleep 5

# ---- Start Celery worker — ONLY the gpu queue ----
echo ">>> Starting Celery GPU worker..."
cd /app

celery -A app.celery_app:celery_app worker \
  --loglevel=info \
  --queues=gpu \
  --hostname=gpu-worker@%h \
  --concurrency=1 \
  --pool=solo
# concurrency=1   → one GPU task at a time (correct for single GPU)
# pool=solo       → avoids fork issues with PyTorch / CUDA in subprocesses