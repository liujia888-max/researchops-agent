#!/usr/bin/env bash
# Start the inference server (bge-m3 + reranker) on autodl-new5, port 8001.
set -euo pipefail

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

cd /root/autodl-tmp
nohup python -u inference_server.py > inference_server.log 2>&1 &
echo "inference server started pid=$!"
