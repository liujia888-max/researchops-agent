#!/usr/bin/env bash
# Start inference server (bge-m3 embedding + lazy reranker) on autodl-new5, port 8001.
# Offline flags make transformers load strictly from the local cache, skipping the
# snapshot completeness check that would otherwise re-fetch ignored .DS_Store files.
set -euo pipefail
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /root/autodl-tmp
pkill -f inference_server.py 2>/dev/null || true
sleep 1
setsid nohup python -u inference_server.py > inference_server.log 2>&1 < /dev/null &
echo "inference server launched pid=$!"
