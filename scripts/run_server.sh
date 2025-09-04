#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Run with Gunicorn and Uvicorn workers
exec gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --workers 4
