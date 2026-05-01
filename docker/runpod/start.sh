#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  Ladybug — RunPod GPU Worker Startup Script
#  Connects to CloudAMQP broker and listens ONLY on the gpu queue
# ════════════════════════════════════════════════════════════════
set -e

echo "══════════════════════════════════════════════"
echo "  Ladybug GPU Worker — RunPod"
echo "══════════════════════════════════════════════"

# ── Show GPU info ────────────────────────────────────────────────
echo ""
echo ">>> GPU:"
nvidia-smi
echo ""

# ── Confirm env vars loaded ──────────────────────────────────────
echo ">>> Broker  : $CELERY_BROKER_URL"
echo ">>> Backend : $CELERY_RESULT_BACKEND"
echo ">>> S3 Bucket: $S3_BUCKET_NAME"
echo ">>> PYTHONPATH: $PYTHONPATH"
echo ""

# ── Brief pause to let network settle after pod start ────────────
echo ">>> Waiting 5s for network..."
sleep 5

# ── Start Celery GPU worker ───────────────────────────────────────
echo ">>> Starting Celery GPU worker..."
cd /app

celery -A app.celery_app:celery_app worker \
  --loglevel=info \
  --queues=gpu \
  --hostname=gpu-worker@%h \
  --concurrency=1 \
  --pool=solo
# concurrency=1  → one GPU task at a time (correct for single GPU)
# pool=solo      → avoids PyTorch/CUDA fork issues in subprocesses