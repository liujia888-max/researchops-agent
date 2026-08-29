#!/usr/bin/env bash
# Start Qdrant server on :6333. Storage defaults to ./storage (under autodl-tmp).
set -euo pipefail
cd /root/autodl-tmp
exec ./qdrant --disable-telemetry
