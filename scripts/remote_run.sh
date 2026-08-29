#!/usr/bin/env bash
# Robust remote startup using screen (survives SSH disconnects on autodl).
# Usage: bash remote_run.sh
set -euo pipefail

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp

# --- 1. inference server (bge-m3 + reranker) on :8001 ---
if screen -ls | grep -q '\.inference'; then
  echo "inference screen already running"
else
  screen -dmS inference bash -c 'python -u inference_server.py > inference_server.log 2>&1'
  echo "inference screen started"
fi

# --- 2. Qdrant download (background, in its own screen) ---
if screen -ls | grep -q '\.qdrant_dl'; then
  echo "qdrant_dl screen already running"
else
  screen -dmS qdrant_dl bash -c 'curl -sL -o qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-unknown-linux-gnu.tar.gz && echo DOWNLOAD_OK'
  echo "qdrant_dl screen started"
fi

sleep 2
screen -ls
