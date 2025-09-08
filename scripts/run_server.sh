#!/usr/bin/env bash
set -e
source venv/bin/activate
# export vars from .env
set -a; source .env; set +a
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level "$LOG_LEVEL"