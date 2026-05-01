#!/bin/bash
set -e

echo "cd /app"
cd /app

echo "Running database migrations..."
# alembic upgrade head   👈 disable this for now

echo "Starting FastAPI..."
exec "$@"
