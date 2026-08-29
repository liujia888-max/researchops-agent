#!/usr/bin/env bash
# Inference server entrypoint — run inside a screen session.
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp
exec python -u inference_server.py
