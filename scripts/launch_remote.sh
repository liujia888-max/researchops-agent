#!/usr/bin/env bash
# Launch remote services fully detached (setsid + nohup) so they survive
# SSH disconnects on autodl (verified: setsid processes outlive the session).
set -euo pipefail

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp

# --- inference server (bge-m3 + reranker) on :8001 ---
setsid nohup bash -c 'cd /root/autodl-tmp && HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com python -u inference_server.py > inference_server.log 2>&1' >/dev/null 2>&1 &
echo "inference launched"

# --- Qdrant download + extract ---
setsid nohup bash -c 'cd /root/autodl-tmp && curl -sL -o qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-unknown-linux-gnu.tar.gz && tar xzf qdrant.tar.gz && echo QDRANT_DOWNLOAD_OK > qdrant.done' >/dev/null 2>&1 &
echo "qdrant download launched"

sleep 2
echo "--- running procs ---"
ps -eo pid,stat,cmd | grep -E 'inference_server|curl.*qdrant' | grep -v grep || true
