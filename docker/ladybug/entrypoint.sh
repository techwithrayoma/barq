#!/bin/bash
set -e

cd /app
export PYTHONPATH=/app/ladybug:$PYTHONPATH


echo "Starting LadyBug ✿ • • •"

# If first argument is celery, run Celery
if [ "$1" = "celery" ]; then
    shift
    exec celery "$@"
else
    # default to CMD
    exec "$@"
fi
