#!/bin/bash
set -e

echo "cd /app"
cd /app

echo "Running database migrations..."
alembic upgrade head

echo "Starting FastAPI..."
exec "$@"
