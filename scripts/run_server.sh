#!/usr/bin/env bash
set -e
source venv/bin/activate
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8002 \
  --workers 4 \
  --log-level info