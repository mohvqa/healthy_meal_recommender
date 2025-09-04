#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Create venv if not exists
if [ ! -d "venv" ]; then
  ${PYTHON:-python3} -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r app/requirements.txt
